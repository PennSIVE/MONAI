import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord
from monai.data import list_data_collate, sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice


def read_data():
    data_root = '/data'
    train_images = sorted(
        glob.glob(os.path.join(data_root, 'imagesTr', '*.nii.gz')))
    train_labels = sorted(
        glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))
    data_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_labels)]
    return data_dicts[:-9], data_dicts[-9:]


def main():
    monai.config.print_config()
    train_files, val_files = read_data()
    train_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.),
                 interp_order=(3, 0), mode='nearest'),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57,
                             a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        # randomly crop out patch samples from big image based on pos / neg ratio
        # the image centers of negative samples must be in valid image area
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', size=(96, 96, 96), pos=1,
                               neg=1, num_samples=4, image_key='image', image_threshold=0),
        # user can also add other random transforms
        # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
        #             rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=['image', 'label'])
    ])
    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.),
                 interp_order=(3, 0), mode='nearest'),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57,
                             a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        ToTensord(keys=['image', 'label'])
    ])

    train_transforms.set_random_state(seed=0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = monai.utils.misc.first(check_loader)
    image, label = (check_data['image'][0][0], check_data['label'][0][0])
    print('image shape: {}, label shape: {}'.format(image.shape, label.shape))
    # plot the slice [:, :, 80]
    plt.figure('check', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.imshow(image[:, :, 80], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('label')
    plt.imshow(label[:, :, 80])
    # plt.show()

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=0.01, num_workers=4
    )
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=list_data_collate)

    val_ds = monai.data.CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=0.01, num_workers=4
    )
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    device = torch.device('cuda:0')
    model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                     strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH)#.to(device)
    loss_function = monai.losses.DiceLoss(to_onehot_y=True, do_softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(600):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, 600))
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            # inputs, labels = batch_data['image'].to(
            #     device), batch_data['label'].to(device)
            inputs, labels = batch_data['image'], batch_data['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print('{}/{}, train_loss: {:.4f}'.format(step,
                                                     len(train_ds) // train_loader.batch_size, loss.item()))
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    # val_inputs, val_labels = val_data['image'].to(
                    #     device), val_data['label'].to(device)
                    val_inputs, val_labels = val_data['image'], val_data['label']
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                             to_onehot_y=True, mutually_exclusive=True)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_metric_model.pth')
                    print('saved new best metric model')
                print('current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}'.format(
                    epoch + 1, metric, best_metric, best_metric_epoch))
    print('train completed, best_metric: {:.4f}  at epoch: {}'.format(best_metric, best_metric_epoch))
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Epoch Average Loss')
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title('Val Mean Dice')
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.show()


main()
