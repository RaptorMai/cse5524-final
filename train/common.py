import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from Dataset.samplers import BalancedClassSampler
from trainer.classifiers import PrototypeClassifier

def train(clip, device, optimizer, model_cfg, train_cfg, train_dataset, eval_dataset, classifer_ckpt_dir):
    # Create balanced sampler
    balanced_sampler = BalancedClassSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'], 
        sampler=balanced_sampler,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True
    )
    ## Config
    epochs = train_cfg['epochs']
    batch_size = train_cfg['batch_size']
    num_batches = len(train_loader)

    print_every_batch = 20
    eval_every = 5
    save_every = 10
    if model_cfg['fine_tune'] == 'LoRA':
        print(f"Train model by {model_cfg['fine_tune']} {model_cfg['LoRA_dim']} dim and classifier head")
    else:
        print(f"Train model by {model_cfg['fine_tune']}")
    for epoch in range(1, epochs + 1):
        clip.model.train()
        print("Epoch : ", epoch)
        loss_arr = []
        i = 0
        progress_bar = tqdm(train_loader, desc='Train Clip')
        for batch_idx, (images, captions, labels, file_paths) in enumerate(progress_bar):
            clip.model.train()
            valid_images = []
            for image in images:
                try:
                    img = Image.open(image)
                    valid_images.append(img)
                except OSError:
                    print(f"Skipping truncated image: {image}")
            # step = i + epoch * batch_size
            i += 1
            with torch.no_grad():
                preprocessed_images = []
                caption_tokens = []
                for idx, (image, caption) in enumerate(zip(valid_images, captions)):
                    # Tokenize the caption
                    caption_token = clip.tokenizer(caption)
                    for _ in range(4):
                        # Preprocess the image
                        preprocessed_image = clip.randomPreprocess(image.convert('RGB'))
                        # Preprocessed images and captions
                        preprocessed_images.append(preprocessed_image)
                        caption_tokens.append(caption_token)
                tensor_caption_tokens = torch.cat(caption_tokens, dim=0).to(device)
                images = torch.stack(preprocessed_images).to(device)
                del preprocessed_images
                del caption_tokens
            with torch.amp.autocast(device):
                logits_per_image, logits_per_text = clip(images, tensor_caption_tokens)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss_i2t = F.cross_entropy(logits_per_image, ground_truth)
            loss_t2i = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_i2t + loss_t2i) / 2
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate sums
            loss_arr.append(loss.cpu().item())

            torch.cuda.empty_cache()
            
            if i % print_every_batch == 0:
                percent_complete = 100 * (i) / num_batches
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]")
                print(f"Contrastive Loss: {loss.item():.4f}")

        loss_arr = torch.tensor(loss_arr, dtype=torch.float32)

        print(f"Avg Loss on Train Epoch {epoch}: {loss_arr.mean():.4f}")

        if epoch % eval_every == 0:
            eval(clip, device, optimizer, model_cfg, train_cfg, train_dataset, eval_dataset)

def train_eval(clip, device, optimizer, model_cfg, train_cfg, eval_dataset, classifer_ckpt_dir):
    # Create balanced sampler
    balanced_sampler = BalancedClassSampler(eval_dataset)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_cfg['batch_size'], 
        sampler=balanced_sampler,
        num_workers=train_cfg.get('num_workers', 4),
        pin_memory=True
    )
    ## Config
    epochs = train_cfg['epochs']
    batch_size = train_cfg['batch_size']
    num_batches = len(eval_loader)

    os.makedirs(classifer_ckpt_dir, exist_ok=True)
    model_save_dir = classifer_ckpt_dir
    os.makedirs(model_save_dir, exist_ok=True)
    if model_cfg['fine_tune'] == 'LoRA':
        model_save_dir = os.path.join(model_save_dir, f"fungi-clef25_bioclip_lora_{model_cfg['LoRA_dim']}")
    else:
        model_save_dir = os.path.join(model_save_dir, f"fungi-clef25_bioclip_fft")
    os.makedirs(model_save_dir, exist_ok=True)

    print_every_batch = 10
    eval_every = 5
    save_every = 10
    if model_cfg['fine_tune'] == 'LoRA':
        print(f"Train model by {model_cfg['fine_tune']} {model_cfg['LoRA_dim']} dim and classifier head")
    else:
        print(f"Train model by {model_cfg['fine_tune']}")
    
    for epoch in range(1, epochs + 1):
        clip.model.train()
        print("Epoch : ", epoch)
        loss_arr = []
        i = 0
        progress_bar = tqdm(eval_loader, desc='Train Clip')
        for batch_idx, (images, captions, labels, file_paths) in enumerate(progress_bar):
            clip.model.train()
            valid_images = []
            for image in images:
                try:
                    img = Image.open(image)
                    valid_images.append(img)
                except OSError:
                    print(f"Skipping truncated image: {image}")
            # step = i + epoch * batch_size
            i += 1
            with torch.no_grad():
                preprocessed_images = []
                caption_tokens = []
                for idx, (image, caption) in enumerate(zip(valid_images, captions)):
                    # Tokenize the caption
                    caption_token = clip.tokenizer(caption)
                    for _ in range(4):
                        # Preprocess the image
                        preprocessed_image = clip.randomPreprocess(image.convert('RGB'))
                        # Preprocessed images and captions
                        preprocessed_images.append(preprocessed_image)
                        caption_tokens.append(caption_token)
                tensor_caption_tokens = torch.cat(caption_tokens, dim=0).to(device)
                images = torch.stack(preprocessed_images).to(device)
                del preprocessed_images
                del caption_tokens
            with torch.amp.autocast(device):
                logits_per_image, logits_per_text = clip(images, tensor_caption_tokens)
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            loss_i2t = F.cross_entropy(logits_per_image, ground_truth)
            loss_t2i = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_i2t + loss_t2i) / 2
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate sums
            loss_arr.append(loss.cpu().item())

            torch.cuda.empty_cache()

            if i % print_every_batch == 0:
                percent_complete = 100 * (i) / num_batches
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]")
                print(f"Contrastive Loss: {loss.item():.4f}")

        loss_arr = torch.tensor(loss_arr, dtype=torch.float32)

        print(f"Avg Loss on Train Epoch {epoch}: {loss_arr.mean():.4f}")

        ## save the model
        if epoch % save_every == 0:
            model_save_path = os.path.join(model_save_dir, f'epoch_{epoch}.pth')
            clip.save(model_save_path)
            print(f"Model saved at {model_save_path}")

def generate_embeddings(dataset, tp, model):
    idxs = np.arange(len(dataset))
    im_names, embs = [], []
    for idx in tqdm(idxs):
        img, text, label, file_path = dataset[idx]

        with torch.no_grad():
            feat = model.extract_img_features(Image.open(img).convert('RGB'))

        im_names.append(os.path.basename(file_path))
        embs.append(feat.detach().cpu().numpy())

    embeddings = pd.DataFrame({'filename': im_names, 'embedding': embs})

    return embeddings

def eval(model, device, optimizer, model_cfg, train_cfg, train_dataset, eval_dataset):
    print("###################### Eval model #######################")
    model.model.eval()
    eval_embeddings = generate_embeddings(eval_dataset, 'test', model)
    eval_dataset.add_embeddings(eval_embeddings)
    train_embeddings = generate_embeddings(train_dataset, 'train', model)
    train_dataset.add_embeddings(train_embeddings)
    print("Eval embeddings generated")
    classifier = PrototypeClassifier(train_dataset, device=device)
    preds, conf = classifier.make_prediction(torch.tensor(np.array(eval_dataset.df.embedding.values.tolist(), dtype=np.float32)))
    preds = preds.cpu().numpy() # [N, 5]
    # Get ground truth from eval dataset instead of train dataset
    eval_ground_truth = np.array(eval_dataset.df['category_id'].values.tolist(), dtype=np.float32) # [N, 1]

    # Calculate Recall@5
    correct = 0
    for i, label in enumerate(eval_ground_truth):
        if label in preds[i]:
            correct += 1

    recall_at_5 = correct / len(eval_ground_truth)
    print(f"Recall@5: {recall_at_5:.4f}")
