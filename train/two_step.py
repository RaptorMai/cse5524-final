import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import os
from torch.cuda.amp import GradScaler, autocast
from Dataset.samplers import BalancedClassSampler

def first_step(clip, device, optimizer, model_cfg, train_cfg, train_dataset):
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
            ## Use original CLIP
            labels = labels.to(device)
            with torch.no_grad():
                preprocessed_images = []
                caption_tokens = []
                for idx, (image, caption) in enumerate(zip(valid_images, captions)):
                    # Tokenize the caption
                    caption_token = clip.tokenizer(caption).to(device)
                    for _ in range(4):
                        # Preprocess the image
                        preprocessed_image = clip.randomPreprocess(image.convert('RGB')).to(device)
                        # Preprocessed images and captions
                        preprocessed_images.append(preprocessed_image)
                        caption_tokens.append(caption_token)
                caption_tokens = torch.cat(caption_tokens, dim=0).to(device)
                images = torch.stack(preprocessed_images).to(device)
            with torch.amp.autocast(device):
                logits_per_image, logits_per_text = clip(images, caption_tokens)
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
            if i % print_every_batch == 0:
                percent_complete = 100 * (i) / num_batches
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]")
                print(f"Contrastive Loss: {loss.item():.4f}")
        loss_arr = torch.tensor(loss_arr)
        print(f"Avg Loss on Train Epoch {epoch}: {loss_arr.mean():.4f}")

def second_step(model, device, optimizer, model_cfg, train_cfg, train_dataset, eval_dataset, classifer_ckpt_dir):
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

    os.makedirs(classifer_ckpt_dir, exist_ok=True)
    model_save_dir = classifer_ckpt_dir
    os.makedirs(model_save_dir, exist_ok=True)
    if model_cfg['fine_tune'] == 'LoRA':
        model_save_dir = os.path.join(model_save_dir, f"fungi-clef25_bioclip_lora_{model_cfg['LoRA_dim']}_Proclassifer")
    else:
        model_save_dir = os.path.join(model_save_dir, f"fungi-clef25_bioclip_fft")
    os.makedirs(model_save_dir, exist_ok=True)

    print_every_batch = 20
    save_every = 10
    print(f"Train model by {model_cfg['fine_tune']} {model_cfg['LoRA_dim']} dim")
    for epoch in range(1, epochs + 1):
        model.train()
        print("Epoch : ", epoch)
        loss_arr = []
        correct_arr = []
        recall_at_5_arr = []

        i = 0
        progress_bar = tqdm(train_loader, desc='Train Clip')
        for batch_idx, (images, captions, labels, file_paths) in enumerate(progress_bar):
            i += 1
            model.train()
            valid_images = []
            for image in images:
                try:
                    img = Image.open(image).convert('RGB')
                    valid_images.append(img)
                except OSError:
                    print(f"Skipping truncated image: {image}")
            ## Use original CLIP
            with torch.no_grad():
                preprocessed_images = []
                new_labels = []
                for idx, (image, label) in enumerate(zip(valid_images, labels)):
                    for _ in range(4):
                        # Preprocess the image
                        preprocessed_image = model.randomPreprocess(image).to(device)
                        # Preprocessed images and captions
                        preprocessed_images.append(preprocessed_image)
                        new_labels.append(label)
                images = torch.stack(preprocessed_images).to(device)
                labels = torch.tensor(new_labels, dtype=torch.long, device=device)
            with torch.amp.autocast(device):
                logits = model(images)
            # Get top-5 predictions
            _, top5_indices = logits.topk(5, dim=1)
            # Check if true label is in top-5 predictions for each sample
            correct_top5 = torch.any(top5_indices == labels.unsqueeze(1), dim=1)
            recall_at_5 = correct_top5.float().mean()

            preds = logits.argmax(dim=1)
            correct = preds == labels
            correct = correct.float().mean()

            loss = F.cross_entropy(logits, labels)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate sums
            loss_arr.append(loss.cpu().item())
            correct_arr.append(correct)
            recall_at_5_arr.append(recall_at_5)

            if i % print_every_batch == 0:
                percent_complete = 100 * (i) / num_batches
                print(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]")
                print(f"Contrastive Loss: {loss.item():.4f}")
                print(f"Top-1 Accuracy: {correct:.4f}")
                print(f"Recall@5: {recall_at_5:.4f}")

        loss_arr = torch.tensor(loss_arr, dtype=torch.float32)
        correct_arr = torch.tensor(correct_arr, dtype=torch.float32)
        recall_at_5_arr = torch.tensor(recall_at_5_arr, dtype=torch.float32)

        print(f"Avg Loss on Train Epoch {epoch}: {loss_arr.mean():.4f}")
        print(f"Avg Top-1 Accuracy on Train Epoch {epoch}: {correct_arr.float().mean():.4f}")
        print(f"Avg Recall@5 on Train Epoch {epoch}: {recall_at_5_arr.float().mean():.4f}")

        ## save the model
        if epoch % save_every == 0:
            eval(model, device, optimizer, model_cfg, train_cfg, eval_dataset)
            model_save_path = os.path.join(model_save_dir, f'epoch_{epoch+80}.pth')
            model.save(model_save_path)
            print(f"Model saved at {model_save_path}")

def eval(model, device, optimizer, model_cfg, train_cfg, eval_dataset):
    print("###################### Eval model #######################")
    eval_loader = DataLoader(eval_dataset, batch_size=train_cfg['batch_size'], shuffle=True)

    ## Config
    batch_size = train_cfg['batch_size']
    num_batches = len(eval_loader)

    print_every_batch = 5
    model.train()

    i = 0
    loss_arr = []
    correct_arr = []
    recall_at_5_arr =[]
    progress_bar = tqdm(eval_loader, desc='EVAL Clip')
    for batch_idx, (images, captions, labels, file_paths) in enumerate(progress_bar):
        i += 1
        model.train()
        valid_images = []
        for image in images:
            try:
                img = Image.open(image).convert('RGB')
                valid_images.append(img)
            except OSError:
                print(f"Skipping truncated image: {image}")
        ## Use original CLIP
        with torch.no_grad():
            preprocessed_images = [model.preprocess(image) for image in valid_images]
            images = torch.stack(preprocessed_images).to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.amp.autocast(device):
            logits = model(images)
        # Get top-5 predictions
        _, top5_indices = logits.topk(5, dim=1)
        # Check if true label is in top-5 predictions for each sample
        correct_top5 = torch.any(top5_indices == labels.unsqueeze(1), dim=1)
        recall_at_5 = correct_top5.float().mean()
        # Store for later averaging
        preds = logits.argmax(dim=1)
        correct = preds == labels
        correct = correct.float().mean()

        loss = F.cross_entropy(logits, labels)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate sums
        loss_arr.append(loss.cpu().item())
        correct_arr.append(correct)
        recall_at_5_arr.append(recall_at_5)
        
        if i % print_every_batch == 0:
            percent_complete = 100 * (i) / num_batches
            print(f"Eval: [{percent_complete:.0f}% {i}/{num_batches}]")
            print(f"Contrastive Loss: {loss.item():.4f}")
            print(f"Top-1 Accuracy: {correct:.4f}")
            print(f"Recall@5: {recall_at_5:.4f}")

    loss_arr = torch.tensor(loss_arr, dtype=torch.float32)
    correct_arr = torch.tensor(correct_arr, dtype=torch.float32)
    recall_at_5_arr = torch.tensor(recall_at_5_arr, dtype=torch.float32)

    print(f"Avg Loss on EVAL: {loss_arr.mean():.4f}")
    print(f"Avg Top-1 Accuracy on EVAL: {correct_arr.float().mean():.4f}")
    print(f"Avg Recall@5 on EVAL: {recall_at_5_arr.float().mean():.4f}")
