from collections import defaultdict
import hydra
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from data import *
from modules import *
from train_segmentation import LitUnsupervisedSegmenter
from crf import dense_crf


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:

    # GPU configuration for using 1 GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(
        torch.cuda.device_count(),
        "GPU available. Name: ",
        torch.cuda.get_device_name(0),
    )
    torch.cuda.empty_cache()

    # print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir

    num_cluster = 12
    experiment_name = "directory_12cluster_date_Mar10_23-50-13"
    checkpoint_name = "epoch=0-step=1200.ckpt"
    result_dir = "../results/predictions/" + experiment_name
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    label_to_color = {
        0: [255, 255, 255],  # White
        1: [0, 0, 255],  # Blue
        2: [0, 255, 0],  # Green
        3: [255, 165, 0],  # Orange
        4: [0, 200, 200],  # Teal
        5: [255, 255, 0],  # Yellow
        6: [128, 0, 0],  # Maroon
        7: [0, 128, 0],  # Dark Green
        8: [255, 0, 255],  # Magenta
        9: [128, 128, 0],  # Olive
        10: [128, 0, 128],  # Purple
        11: [0, 128, 128],  # Aqua
        12: [255, 128, 0],  # Light Orange
        13: [128, 255, 0],  # Lime
        14: [255, 0, 128],  # Pink
        15: [0, 255, 128],  # Mint
        16: [128, 0, 255],  # Violet
        17: [0, 128, 255],  # Sky Blue
        18: [255, 128, 128],  # Coral
        19: [128, 255, 128],  # Light Green
        20: [128, 128, 255],  # Periwinkle
        21: [255, 255, 128],  # Pale Yellow
        22: [255, 128, 255],  # Lavender
        23: [128, 128, 128],  # Gray
    }

    full_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name="directory",
        crop_type=None,
        image_set="all",
        transform=get_transform(320, False, "center"),
        target_transform=get_transform(320, True, "center"),
        cfg=cfg,
    )

    test_loader = DataLoader(
        full_dataset,
        20,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )

    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        "D:/waglar/checkpoints/ahrtal/" + experiment_name + "/" + checkpoint_name
    )
    print(OmegaConf.to_yaml(model.cfg))
    model.eval().cuda()
    par_model = torch.nn.DataParallel(model.net)

    outputs = defaultdict(list)
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if i > 209:  # 210*20 = 4200 is the number of images in the ahrtal dataset
                break

            img = batch["img"]  # .cuda()
            label = batch["label"].cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(
                code, label.shape[-2:], mode="bilinear", align_corners=False
            )
            cluster_prob = model.cluster_probe(code, 2, log_probs=True)
            cluster_pred = cluster_prob.argmax(1)

            model.test_cluster_metrics.update(cluster_pred, label)

            outputs["img"].append(img.cpu())
            # outputs["label"].append(label.cpu())
            # outputs["cluster_pred"].append(cluster_pred.cpu())
            outputs["cluster_prob"].append(cluster_prob.cpu())
    model.test_cluster_metrics.compute()

    # img_num = 41
    # output = {
    #     k: torch.cat(v, dim=0)[10 * 10 * img_num : 10 * 10 * (img_num + 1)]
    #     for k, v in outputs.items()
    # }

    # full_image = (
    #     output["img"]
    #     .reshape(10, 10, 3, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(3, 320 * 10, 320 * 10)
    # )

    # full_cluster_prob = (
    #     output["cluster_prob"]
    #     .reshape(10, 10, 5, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(5, 320 * 10, 320 * 10)
    # )

    # crf_probs = dense_crf(full_image.cpu().detach(), full_cluster_prob.cpu().detach())
    # no_crf_probs = full_cluster_prob.numpy()

    # reshaped_img = unnorm(full_image).permute(1, 2, 0)

    # reshaped_preds_crf = model.test_cluster_metrics.map_clusters(
    #     np.expand_dims(crf_probs.argmax(0), 0)
    # )
    # reshaped_preds_no_crf = model.test_cluster_metrics.map_clusters(
    #     np.expand_dims(no_crf_probs.argmax(0), 0)
    # )

    # fig, ax = plt.subplots(1, 3, figsize=(4 * 3, 4))
    # ax[0].imshow(reshaped_img)
    # ax[1].imshow(reshaped_preds_crf)
    # ax[2].imshow(reshaped_preds_no_crf)

    # remove_axes(ax)
    # plt.show()

    # saving every image:
    for img_num in range(42):
        output = {
            k: torch.cat(v, dim=0)[10 * 10 * img_num : 10 * 10 * (img_num + 1)]
            for k, v in outputs.items()
        }

        full_image = (
            output["img"]
            .reshape(10, 10, 3, 320, 320)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, 320 * 10, 320 * 10)
        )

        full_cluster_prob = (
            output["cluster_prob"]
            .reshape(10, 10, num_cluster, 320, 320)
            .permute(2, 0, 3, 1, 4)
            .reshape(num_cluster, 320 * 10, 320 * 10)
        )

        crf_probs = dense_crf(
            full_image.cpu().detach(), full_cluster_prob.cpu().detach()
        )
        no_crf_probs = full_cluster_prob.numpy()

        reshaped_img = unnorm(full_image).permute(1, 2, 0)
        reshaped_preds_crf = model.test_cluster_metrics.map_clusters(
            np.expand_dims(crf_probs.argmax(0), 0)
        )
        reshaped_preds_no_crf = model.test_cluster_metrics.map_clusters(
            np.expand_dims(no_crf_probs.argmax(0), 0)
        )

        # Image.fromarray(reshaped_img).save(
        #     join(join(result_dir, "img", str(img_num) + ".png"))
        # )
        # Image.fromarray(reshaped_preds_crf).save(
        #     join(join(result_dir, "cluster", str(img_num) + "_crf.png"))
        # )
        # Image.fromarray(reshaped_preds_no_crf).save(
        #     join(join(result_dir, "cluster", str(img_num) + "_no_crf.png"))
        # )

        # print("Saved image number ", img_num)

        reshaped_img = reshaped_img.numpy() * 255
        reshaped_img = reshaped_img.astype(np.uint8)
        reshaped_preds_crf = reshaped_preds_crf.numpy()
        reshaped_preds_crf = reshaped_preds_crf.astype(np.uint8)
        reshaped_preds_no_crf = reshaped_preds_no_crf.numpy()
        reshaped_preds_no_crf = reshaped_preds_no_crf.astype(np.uint8)

        h, w = reshaped_preds_crf.shape
        reshaped_preds_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for gray, rgb in label_to_color.items():
            reshaped_preds_crf_rgb[reshaped_preds_crf == gray, :] = rgb
        reshaped_preds_no_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for gray, rgb in label_to_color.items():
            reshaped_preds_no_crf_rgb[reshaped_preds_no_crf == gray, :] = rgb

        Image.fromarray(reshaped_img).save(
            join(join(result_dir, "img", str(img_num) + ".png"))
        )
        Image.fromarray(reshaped_preds_crf_rgb).save(
            join(join(result_dir, "cluster", str(img_num) + "_crf.png"))
        )
        Image.fromarray(reshaped_preds_no_crf_rgb).save(
            join(join(result_dir, "cluster", str(img_num) + "_no_crf.png"))
        )

        print("Saved image number ", img_num + 1, " of 42.")


if __name__ == "__main__":
    prep_args()
    my_app()
