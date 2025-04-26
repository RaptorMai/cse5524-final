import torch
import os
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import OrderedDict
from engine.optimizer import make_optimizer
from utils.misc import AverageMeter, EarlyStop
from utils.setup_logging import get_logger
from timm.utils import accuracy, update_summary
import numpy as np
from collections import defaultdict
from loss.seasaw import SeesawLoss
from loss.focal import FocalLoss
from loss.siglip2 import SigLIP2Loss
from torch.cuda.amp import GradScaler
import xgboost as xgb
from xgboost import XGBClassifier
logger = get_logger("PETL_vision")

class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """

    def __init__(
            self,
            model, tune_parameters, params
    ) -> None:
        self.params = params
        self.model = model
        self.device = params.device
        # if params.xgboost_path is not None and os.path.exists(params.xgboost_path):
        #     self.xg = xgb.XGBRegressor()
        #     logger.info(f"Loading xgboost model from {params.xgboost_path}")
        #     self.xg.load_model(f'{params.xgboost_path}')

        if params.loss == 'cross_entropy':
            self.cls_criterion = nn.CrossEntropyLoss()
        elif params.loss == 'focal_loss':
            self.cls_criterion = FocalLoss(gamma=2.0, alpha=0.25,
                use_sigmoid=False, reduction='mean', loss_weight=1.0, num_classes=params.class_num)
        elif params.loss == 'seesaw_loss':
            self.cls_criterion = SeesawLoss(num_classes=params.class_num, p=0.8, q=2.0, reduction='mean')
        elif params.loss == 'siglip2_loss':
            self.cls_criterion = SigLIP2Loss(temperature=0.15, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {params.loss}")

        if 'test_data' not in params:
            # solver related
            logger.info("\tSetting up the optimizer...")
            self.optimizer = make_optimizer(tune_parameters, params)
            self.scheduler = CosineLRScheduler(self.optimizer, t_initial=params.epoch,
                                               warmup_t=params.warmup_epoch, lr_min=params.lr_min,
                                               warmup_lr_init=params.warmup_lr_init)
            self.scaler = GradScaler()
            self.total_epoch = self.params.epoch
            if self.params.early_patience > 0:
                self.early_stop_check = EarlyStop(self.params.early_patience)

    def forward_one_batch(self, samples, targets, is_train, prototype=None):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            samples
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        samples = samples.to(self.device, non_blocking=True)  # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        # forward
        with torch.set_grad_enabled(is_train):
            if prototype is not None:
                embedding = self.model.forward_features(samples)  # (batchsize, D)
                if embedding.ndim == 3:
                    embedding = embedding[:, 0, :]
                embedding = nn.functional.normalize(embedding, dim=1)  # (batchsize, D)
                outputs = embedding @ prototype.T  # (batchsize, num_cls)
            else:
                outputs = self.model(samples)  # (batchsize, num_cls)
                    # if 'test_data' in self.params and self.params.test_data == 'eval_imagenet-r':
                    #     outputs = outputs[:, R_CLASS_SUBLIST_MASK]
                    # elif 'test_data' in self.params and self.params.test_data == 'eval_imagenet-a':
                    #     outputs = outputs[:, A_CLASS_SUBLIST_MASK]
            loss = self.cls_criterion(outputs, targets)
            torch.cuda.empty_cache()

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1, (-1, -1)
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1, (-1, -1)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # loss.backward()
            # self.optimizer.step()

        return loss, outputs, (acc1, acc5)

    def train_one_epoch(self, epoch, loader):
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        lr = self.scheduler._get_lr(epoch)
        logger.info(
            "Training {} / {} epoch, with learning rate {}".format(
                epoch + 1, self.total_epoch, lr
            )
        )
        # Enable training mode
        self.model.train()

        num_updates = epoch * len(loader)
        for idx, (samples, targets) in enumerate(loader):
            train_loss, _, (acc1, acc5) = self.forward_one_batch(samples, targets, True)
            if not isinstance(train_loss, int):
                loss_m.update(train_loss.item(), samples.shape[0])
                top1_m.update(acc1.item(), samples.shape[0])
                top5_m.update(acc5.item(), samples.shape[0])
            del train_loss, acc1, acc5, _, samples, targets
            num_updates += 1
            self.scheduler.step_update(num_updates=num_updates, metric=loss_m.avg)
            torch.cuda.empty_cache()

        logger.info(
            "Epoch {} / {}: ".format(epoch + 1, self.total_epoch)
            + "average train loss: {:.2f}, ".format(loss_m.avg)
            + "average train top1: {:.2f} ".format(top1_m.avg)
            + "average train top5: {:.2f}".format(top5_m.avg))

        return OrderedDict(
            [('loss', round(loss_m.avg, 2)), ('top1', round(top1_m.avg, 2)), ('top5', round(top5_m.avg, 2))])

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """

        train_loader.dataset.set_mode('train')
        for epoch in range(self.total_epoch):

            train_metrics = self.train_one_epoch(epoch, train_loader)

            if (epoch % self.params.eval_freq == 0) or epoch == self.total_epoch - 1:
                if val_loader is not None:
                    eval_metrics = self.eval_classifier(val_loader, "val")
                    # eval_metrics_prototypes = self.eval_classifier_with_prototypes(train_loader, val_loader)
                else:
                    raise Exception('Both val and test loaders are missing. ')

                if self.params.early_patience > 0:
                    stop, save_model = self.early_stop_check.early_stop(eval_metrics)
                    if save_model and self.params.store_ckp:
                        torch.save({'model_state_dict': self.model.state_dict()},
                                   os.path.join(self.params.output_dir, 'model.pt'))
                    if stop:
                        return train_metrics, self.early_stop_check.max_metrics, eval_metrics
                if self.params.debug:
                    update_summary(
                        epoch, train_metrics, eval_metrics, os.path.join(self.params.output_dir, 'summary.csv'),
                        write_header=epoch == 0)
            self.scheduler.step(epoch)

        if self.params.store_ckp:
            torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.params.model_dir, 'model.pt'))
            logger.info("Model saved to {}".format(os.path.join(self.params.model_dir, 'model.pt')))

        return train_metrics, self.early_stop_check.max_metrics, eval_metrics

    @torch.no_grad()
    def build_prototypes(self, train_loader):
        train_loader.dataset.set_mode('eval')
        self.model.eval()
        embeddings_by_class = defaultdict(list)

        with torch.no_grad():
            for image, label in train_loader.dataset:
                image = image.unsqueeze(0).to(self.device)
                feature = self.model.forward_features(image)         # extract features
                # embeddings = model.forward_head(features, pre_logits=True)  # optionally skip final head if needed
                embeddings_by_class[label].append(feature)
        # Compute mean embedding per class
        prototypes = {}
        for label, features_list in embeddings_by_class.items():
            features = torch.cat(features_list, dim=0)  # Concatenate all features for this class
            if features.ndim == 3:
                global_features = features[:, 0, :]
                mean_global_features = global_features.mean(dim=0)
            elif features.ndim == 2:
                global_features = features
                mean_global_features = features.mean(dim=0)
            # Compute distances from each feature to the current prototype
            distances = torch.norm(global_features - mean_global_features.unsqueeze(0), dim=1)  # Euclidean distance

            # Select the 50% closest features
            num_to_keep = (len(features) + 1) // 2
            closest_indices = torch.argsort(distances)[:num_to_keep]
            closest_features = features[closest_indices]

            # Recompute the prototype using only the closest features
            prototype = closest_features.mean(dim=0)
            prototypes[label] = prototype
        embeddings_by_class 
        # Concatenate all prototype vectors into a single tensor
        prototype = torch.stack([prototypes[label] for label in sorted(prototypes.keys())], dim=0).to(self.device)
        del prototypes
        # Normalize the prototype vectors
        if prototype.ndim == 3:
            prototype = prototype[:, 0, :]
        prototype = nn.functional.normalize(prototype, dim=1)
        train_loader.dataset.set_mode('train')
        return prototype

    @torch.no_grad()
    def eval_classifier_with_prototypes(self, train_loader, val_loader):
        """evaluate classifier with prototypes"""
        # Build prototypes
        prototype = self.build_prototypes(train_loader)

        # Evaluate on validation set
        val_metrics = self.eval_classifier(val_loader, "val_prototypes", prototype=prototype)

        return val_metrics

    @torch.no_grad()
    def eval_classifier(self, loader, prefix, prototype=None):
        """evaluate classifier"""

        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        # Enable eval mode
        self.model.eval()
        loader.dataset.set_mode('eval')

        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(loader):
                loss, outputs, (acc1, acc5) = self.forward_one_batch(samples, targets, False, prototype=prototype)
                if not isinstance(loss, int):
                    loss_m.update(loss.item(), samples.shape[0])
                    top1_m.update(acc1.item(), samples.shape[0])
                    top5_m.update(acc5.item(), samples.shape[0])
                del loss, outputs, acc1, acc5
            del prototype
        logger.info(
            f"Inference ({prefix}):"
            + "average loss: {:.2f}, ".format(loss_m.avg)
            + "average top1: {:.2f} ".format(top1_m.avg)
            + "average top5: {:.2f}".format(top5_m.avg))
        loader.dataset.set_mode('train')
        return OrderedDict(
            [('loss', round(loss_m.avg, 2)), ('top1', round(top1_m.avg, 2)), ('top5', round(top5_m.avg, 2))])
    
    @torch.no_grad()
    def test_classifier(self, train_loader, test_loader, test_method, sub_path, params):
        """evaluate classifier with prototypes"""
        train_loader.dataset.set_mode('train')
        test_loader.dataset.set_mode('eval')
        # Build prototypes
        if test_method == 'prototype':
            ############### test with prototypes ###############
            
            if params.xgboost_path is not None and os.path.exists(params.xgboost_path):
                booster = xgb.Booster()
                booster.load_model(f'{params.xgboost_path}')
                logger.info(f"Loading xgboost model from {params.xgboost_path}")
                train_feature_cols, X_train = train_loader.dataset.prepare_xg()
                test_feature_cols, X_test = test_loader.dataset.prepare_xg(train_feature_cols)
                dtrain = train_loader.dataset.dtrain
                dtest = test_loader.dataset.dtest
                train_leaf_ids = booster.predict(dtrain, pred_leaf=True) # shape (n_train, n_trees)
                # train_leaf_ids = self.xg.apply(X_train).astype(int)
                leaf_to_train_cls = defaultdict(set)
                for train_idx, leaf in enumerate(train_leaf_ids):
                    leaf_to_train_cls[leaf].add(train_loader.dataset.df.iloc[train_idx]['category_id'])
                test_leaf_ids = booster.predict(dtest, pred_leaf=True)  # shape (n_test, n_trees)
                # test_leaf_ids = self.xg.apply(X_test).astype(int)  # shape (n_test,)
            
            logger.info("Encoding test images for prototype...")
            embeddings = []
            for idx, (samples, _) in enumerate(test_loader):
                samples = samples.to(self.device, non_blocking=True)  # (batchsize, 2048)
                embedding = self.model.forward_features(samples)  # (batchsize, D)
                if embedding.ndim == 3:
                    embedding = embedding[:, 0, :]
                embeddings.append(embedding.cpu())
            embeddings = torch.cat(embeddings, dim=0).to(self.device)  # (batchsize, D)
            embeddings = nn.functional.normalize(embeddings, dim=1)  # (batchsize, D)
            prototype = self.build_prototypes(train_loader)
            logits = torch.matmul(embeddings, prototype.T)
        elif test_method == 'classifier':
            logits = []
            logger.info("Encoding test images for classifier...")
            for idx, (samples, _) in enumerate(test_loader):
                samples = samples.to(self.device, non_blocking=True)  # (batchsize, 2048)
                logits.append(self.model(samples).cpu())
            logits = torch.cat(logits, dim=0).to(self.device)
        else:
            raise ValueError(f"Unknown test method: {test_method}")
        if params.xgboost_path is not None:
            preds = []  # will be list of length n_test, each a list of 5 class‑ids
            n_test, n_cls = logits.shape

            for i in range(n_test):
                # 1) get the candidate classes from the leaf
                leaf = test_leaf_ids[i]
                leaf_classes = list(leaf_to_train_cls[leaf])   # e.g. [12, 45, 78, …]

                # 2) score only those leaf classes
                leaf_scores = logits[i, leaf_classes]          # shape = (len(leaf_classes),)

                # 3) pick the top k1 within the leaf
                k1 = min(len(leaf_classes), 5)
                top1_scores, top1_idx = torch.topk(leaf_scores, k1)
                top1_cls = [ leaf_classes[j] for j in top1_idx.tolist() ]

                # 4) if fewer than 5, pad with the global top scores (excluding already chosen)
                if k1 < 5:
                    k2 = 5 - k1
                    full_scores = logits[i].clone()

                    # mask out the ones we’ve already selected
                    mask = torch.ones(n_cls, dtype=torch.bool, device=full_scores.device)
                    mask[top1_cls] = False
                    full_scores[~mask] = float('-inf')

                    # pick the next-best k2 globally
                    top2_scores, top2_idx = torch.topk(full_scores, k2)
                    top2_cls = top2_idx.tolist()
                    top1_cls.extend(top2_cls)
                preds.append(top1_cls)
        else:
            preds = torch.topk(logits, 5, dim=1).indices
            preds = preds.cpu().numpy().tolist()

        # Save the predictions
        test_dataset = test_loader.dataset
        test_dataset.df["top5_preds"] = [" ".join(map(str, pred_list)) for pred_list in preds]
        # Group by observationID and preserve top-5 predictions
        submission = (
            test_dataset.df.groupby("observationID")["top5_preds"]
            .first()  # Take the first top-5 prediction set for each observationID
            .reset_index()
        )
        
        # Rename columns for submission format
        submission = submission.rename(columns={"observationID": "observationId", "top5_preds": "predictions"})
        submission = submission.drop_duplicates(subset="observationId")
        submission.to_csv(sub_path, index=None)
        logger.info(f"Test results saved to {sub_path}")

    def load_weight(self, model_path=None):
        self.model.load_state_dict(torch.load(self.params.model_dir + '/model.pt')['model_state_dict'])

    @torch.no_grad()
    def collect_logits(self, loader):
        self.model.eval()
        all_logits = []
        gt = []
        with torch.no_grad():
            for batch_idx, (samples, targets) in enumerate(loader):
                loss, outputs, (acc1, acc5) = self.forward_one_batch(samples, targets, False)
                all_logits.append(outputs.cpu().detach().numpy())
                gt.append(targets.cpu().detach().numpy())
        return np.concatenate(all_logits, axis=0), np.concatenate(gt, axis=0)