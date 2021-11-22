from torchvision.utils import make_grid


def vis_results(split, log_writer, epoch, nrow, input_imgs, mean_shape_pred, shape_light_frontal, shape_light_pred,
                uv_images_pred):
    imgs_grid = make_grid(input_imgs, nrow=nrow, padding=0)
    log_writer.add_image(tag=f'{split}/Ground truth', img_tensor=imgs_grid, global_step=epoch)

    # masks_grid = make_grid(mask_pred[:int(args.batch_size / 2), None, :, :], nrow=nrow, padding=0)
    # log_writer.add_image(tag='Train/Masks', img_tensor=masks_grid, global_step=epoch)

    mean_shape_grid = make_grid(mean_shape_pred, nrow=nrow, padding=0)
    log_writer.add_image(tag=f'{split}/Mean Shape', img_tensor=mean_shape_grid, global_step=epoch)

    if shape_light_frontal is not None:
        shapes_light_frontal_grid = make_grid(shape_light_frontal, nrow=nrow, padding=0)
        log_writer.add_image(tag=f'{split}/Shapes frontal light', img_tensor=shapes_light_frontal_grid,
                             global_step=epoch)

    if shape_light_pred is not None:
        shapes_light_pred_grid = make_grid(shape_light_pred, nrow=nrow, padding=0)
        log_writer.add_image(tag=f'{split}/Shapes predicted light', img_tensor=shapes_light_pred_grid,
                             global_step=epoch)

    if uv_images_pred is not None:
        uv_images_grid = make_grid(uv_images_pred, nrow=nrow, padding=0)
        log_writer.add_image(tag=f'{split}/Textures', img_tensor=uv_images_grid, global_step=epoch)
