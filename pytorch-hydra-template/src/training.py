import numpy as np
import torch
import os
import wandb
from tqdm import tqdm
from utils import label_accuracy_score, add_hist

categroies = [
    "Backgroud",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


def validation(epoch, model, data_loader, criterion, device):
    print(f"Start validation #{epoch}")
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):

            images = torch.stack(images)
            masks = torch.stack(masks).long()

            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)

        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [
            {classes: round(IoU, 4)} for IoU, classes in zip(IoU, categroies)
        ]

        avrg_loss = total_loss / cnt
        print(
            f"Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}"
        )
        val_dict = {
            "val/loss": loss.item(),
            "val/acc": acc,
            "val/acc_cls": acc_cls,
            "val/mIoU": mIoU,
            "val/fwavacc": fwavacc,
            "val/IoU_Backgroud": IoU[0],
            "val/IoU_General trash": IoU[1],
            "val/IoU_Paper": IoU[2],
            "val/IoU_Paper pack": IoU[3],
            "val/IoU_Metal": IoU[4],
            "val/IoU_Glass": IoU[5],
            "val/IoU_Plastic": IoU[6],
            "val/IoU_Styrofoam": IoU[7],
            "val/IoU_Plastic bag": IoU[8],
            "val/IoU_Battery": IoU[9],
            "val/IoU_Clothing": IoU[10],
        }
        wandb.log(val_dict)
        print(f"IoU by class : {IoU_by_class}")

    return avrg_loss, mIoU


def save_model(model, saved_dir, file_name="efficient_unet_best_model.pt"):
    check_point = {"net": model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def train(
    model,
    data_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    saved_dir,
    val_every,
    log_step,
    exp_name,
    save_every_epoch: bool,
):
    print(f"Start training..")
    n_class = 11
    best_loss = 9999999
    best_metric = 0
    scaler = torch.cuda.amp.GradScaler()

    # save directory checking
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if not os.path.exists(os.path.join(saved_dir, exp_name)):
        os.makedirs(os.path.join(saved_dir, exp_name))

    for epoch in range(num_epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            # inference
            with torch.cuda.amp.autocast():
                outputs = model(images)
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)

            # loss.backward()
            scaler.scale(loss).backward()  # gradient 계산
            # optimizer.step()
            scaler.step(optimizer)  # 모델 업데이트
            scaler.update()
            scheduler.step()

            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력 & wandb logging
            if (step + 1) % log_step == 0:
                lr_dict = {"optimize/learning_rate": scheduler.get_last_lr()[0]}
                wandb.log(lr_dict)
                train_dict = {
                    "train/loss": loss.item(),
                    "train/acc": acc,
                    "train/acc_cls": acc_cls,
                    "train/mIoU": mIoU,
                    "train/fwavacc": fwavacc,
                    "train/IoU_Backgroud": IoU[0],
                    "train/IoU_General trash": IoU[1],
                    "train/IoU_Paper": IoU[2],
                    "train/IoU_Paper pack": IoU[3],
                    "train/IoU_Metal": IoU[4],
                    "train/IoU_Glass": IoU[5],
                    "train/IoU_Plastic": IoU[6],
                    "train/IoU_Styrofoam": IoU[7],
                    "train/IoU_Plastic bag": IoU[8],
                    "train/IoU_Battery": IoU[9],
                    "train/IoU_Clothing": IoU[10],
                }
                wandb.log(train_dict)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], \
                Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}"
        )
        if save_every_epoch:
            print(f"Save model in {saved_dir+exp_name}")
            save_model(model, saved_dir + "/" + exp_name, f"epoch_{epoch+1}.pt")

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU = validation(
                epoch + 1, model, val_loader, criterion, device
            )
            # best loss model save
            if avrg_loss < best_loss:
                print(f"Best Loss at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir+exp_name}")
                best_loss = avrg_loss
                save_model(model, saved_dir + "/" + exp_name, "best_loss.pt")
            # best metric model save
            if mIoU > best_metric:
                print(f"Best mIoU at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir+exp_name}")
                best_metric = mIoU
                save_model(model, saved_dir + "/" + exp_name, "best_mIoU.pt")
