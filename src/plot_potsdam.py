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

    result_dir = "../results/predictions/potsdam_fusion1_date_Apr25_16-29-51"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    label_to_color = {
        0: [255, 255, 255],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [255, 165, 0],
        4: [0, 200, 200],
        5: [255, 255, 0],
    }

    full_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name="potsdamraw",
        crop_type=None,
        image_set="all",
        transform=get_transform(320, False, "center"),
        target_transform=get_transform(320, True, "center"),
        cfg=cfg,
    )

    test_loader = DataLoader(
        full_dataset,
        32,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )

    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        "D:/waglar/checkpoints/potsdam/potsdam_fusion1_date_Apr25_16-29-51/epoch=12-step=10800.ckpt"
    )
    print(OmegaConf.to_yaml(model.cfg))
    model.eval().cuda()
    par_model = torch.nn.DataParallel(model.net)

    outputs = defaultdict(list)
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if i > 267:
                break

            if hasattr(model.cfg, "fusion_type"):
                if (
                    model.cfg.fusion_type == "fusion0"
                    or model.cfg.fusion_type == "fusion1"
                ):
                    img = batch["img1"]
                    img2 = batch["img2"]
                elif (
                    model.cfg.fusion_type == "fusion2"
                    or model.cfg.fusion_type == "fusion3"
                ):
                    img = batch["img"]
                    ndsm = batch["ndsm"]
                else:
                    img = batch["img"]
            else:
                img = batch["img"]
            label = batch["label"].cuda()

            if hasattr(model.cfg, "fusion_type"):
                if model.cfg.fusion_type == "fusion0":
                    feats1, code1_1 = par_model(img)
                    feats2, code1_2 = par_model(img2)
                    feats, code2_1 = par_model(img.flip(dims=[3]))
                    feats, code2_2 = par_model(img2.flip(dims=[3]))
                    code1 = code1_1 + code1_2
                    code2 = code2_1 + code2_2
                elif model.cfg.fusion_type == "fusion1":
                    feats, code1 = par_model(img, img2)
                    feats, code2 = par_model(img.flip(dims=[3]), img2.flip(dims=[3]))
                elif (
                    model.cfg.fusion_type == "fusion2"
                    or model.cfg.fusion_type == "fusion3"
                ):
                    feats, code1 = par_model(img, ndsm)
                    feats, code2 = par_model(img.flip(dims=[3]), ndsm.flip(dims=[3]))
                else:
                    feats, code1 = par_model(img)
                    feats, code2 = par_model(img.flip(dims=[3]))
            else:
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
            outputs["label"].append(label.cpu())
            outputs["cluster_pred"].append(cluster_pred.cpu())
            outputs["cluster_prob"].append(cluster_prob.cpu())
    model.test_cluster_metrics.compute()

    # img_num = 6
    # outputs = {
    #     k: torch.cat(v, dim=0)[15 * 15 * img_num : 15 * 15 * (img_num + 1)]
    #     for k, v in outputs.items()
    # }

    # full_image = (
    #     outputs["img"]
    #     .reshape(15, 15, 3, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(3, 320 * 15, 320 * 15)
    # )

    # full_cluster_prob = (
    #     outputs["cluster_prob"]
    #     .reshape(15, 15, 3, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(3, 320 * 15, 320 * 15)
    # )

    # # crf_probs = dense_crf(full_image.cpu().detach(),
    # #                       full_cluster_prob.cpu().detach())
    # crf_probs = full_cluster_prob.numpy()
    # print(crf_probs.shape)

    # reshaped_label = (
    #     outputs["label"]
    #     .reshape(15, 15, 320, 320)
    #     .permute(0, 2, 1, 3)
    #     .reshape(320 * 15, 320 * 15)
    # )
    # reshaped_img = unnorm(full_image).permute(1, 2, 0)
    # reshaped_preds = model.test_cluster_metrics.map_clusters(
    #     np.expand_dims(crf_probs.argmax(0), 0)
    # )

    # fig, ax = plt.subplots(1, 3, figsize=(4 * 3, 4))
    # ax[0].imshow(reshaped_img)
    # ax[1].imshow(reshaped_preds)
    # ax[2].imshow(reshaped_label)

    # Image.fromarray(reshaped_img.cuda()).save(
    #     join(join(result_dir, "img", str(img_num) + ".png"))
    # )
    # Image.fromarray(reshaped_preds).save(
    #     join(join(result_dir, "cluster", str(img_num) + ".png"))
    # )

    # remove_axes(ax)
    # plt.show()

    # # plotting a specific image:
    # img_num = 0
    # print("Now plotting image number ", img_num, "...")
    # output = {
    #     k: torch.cat(v, dim=0)[15 * 15 * img_num : 15 * 15 * (img_num + 1)]
    #     for k, v in outputs.items()
    # }

    # full_image = (
    #     output["img"]
    #     .reshape(15, 15, 3, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(3, 320 * 15, 320 * 15)
    # )

    # # 3 cluster channels:
    # full_cluster_prob = (
    #     output["cluster_prob"]
    #     .reshape(15, 15, 3, 320, 320)
    #     .permute(2, 0, 3, 1, 4)
    #     .reshape(3, 320 * 15, 320 * 15)
    # )

    # # # 6 cluster channels:
    # # full_cluster_prob = (
    # #     output["cluster_prob"]
    # #     .reshape(15, 15, 6, 320, 320)
    # #     .permute(2, 0, 3, 1, 4)
    # #     .reshape(6, 320 * 15, 320 * 15)
    # # )

    # crf_probs = dense_crf(full_image.cpu().detach(), full_cluster_prob.cpu().detach())
    # no_crf_probs = full_cluster_prob.numpy()

    # reshaped_label = (
    #     output["label"]
    #     .reshape(15, 15, 320, 320)
    #     .permute(0, 2, 1, 3)
    #     .reshape(320 * 15, 320 * 15)
    # )
    # reshaped_img = unnorm(full_image).permute(1, 2, 0)
    # reshaped_preds_crf = model.test_cluster_metrics.map_clusters(
    #     np.expand_dims(crf_probs.argmax(0), 0)
    # )
    # reshaped_preds_no_crf = model.test_cluster_metrics.map_clusters(
    #     np.expand_dims(no_crf_probs.argmax(0), 0)
    # )

    # # model.test_cluster_metrics.update(reshaped_preds_crf, reshaped_label)
    # # crf_acc = model.test_cluster_metrics.compute()
    # # print(crf_acc)

    # h, w = reshaped_preds_crf.shape
    # reshaped_label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # for gray, rgb in label_to_color.items():
    #     reshaped_label_rgb[reshaped_label == gray, :] = rgb
    # reshaped_preds_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # for gray, rgb in label_to_color.items():
    #     reshaped_preds_crf_rgb[reshaped_preds_crf == gray, :] = rgb
    # reshaped_preds_no_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # for gray, rgb in label_to_color.items():
    #     reshaped_preds_no_crf_rgb[reshaped_preds_no_crf == gray, :] = rgb

    # fig, ax = plt.subplots(1, 4, figsize=(4 * 3, 4))
    # ax[0].imshow(reshaped_img)
    # ax[1].imshow(reshaped_preds_crf_rgb)
    # ax[2].imshow(reshaped_preds_no_crf_rgb)
    # ax[3].imshow(reshaped_label_rgb)

    # print("Showing image number ", img_num)
    # remove_axes(ax)
    # plt.show()

    # saving every image:
    for i in range(38):
        img_num = i
        output = {
            k: torch.cat(v, dim=0)[15 * 15 * img_num : 15 * 15 * (img_num + 1)]
            for k, v in outputs.items()
        }

        full_image = (
            output["img"]
            .reshape(15, 15, 3, 320, 320)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, 320 * 15, 320 * 15)
        )

        # 3 cluster channels:
        full_cluster_prob = (
            output["cluster_prob"]
            .reshape(15, 15, 3, 320, 320)
            .permute(2, 0, 3, 1, 4)
            .reshape(3, 320 * 15, 320 * 15)
        )

        # # 6 cluster channels:
        # full_cluster_prob = (
        #     output["cluster_prob"]
        #     .reshape(15, 15, 6, 320, 320)
        #     .permute(2, 0, 3, 1, 4)
        #     .reshape(6, 320 * 15, 320 * 15)
        # )

        crf_probs = dense_crf(
            full_image.cpu().detach(), full_cluster_prob.cpu().detach()
        )
        no_crf_probs = full_cluster_prob.numpy()

        # reshaped_label = (
        #     output["label"]
        #     .reshape(15, 15, 320, 320)
        #     .permute(0, 2, 1, 3)
        #     .reshape(320 * 15, 320 * 15)
        # )
        # reshaped_img = unnorm(full_image).permute(1, 2, 0)
        reshaped_preds_crf = model.test_cluster_metrics.map_clusters(
            np.expand_dims(crf_probs.argmax(0), 0)
        )
        reshaped_preds_no_crf = model.test_cluster_metrics.map_clusters(
            np.expand_dims(no_crf_probs.argmax(0), 0)
        )

        # reshaped_img = reshaped_img.numpy() * 255
        # reshaped_img = reshaped_img.astype(np.uint8)
        reshaped_preds_crf = reshaped_preds_crf.numpy()
        reshaped_preds_crf = reshaped_preds_crf.astype(np.uint8)
        reshaped_preds_no_crf = reshaped_preds_no_crf.numpy()
        reshaped_preds_no_crf = reshaped_preds_no_crf.astype(np.uint8)
        # reshaped_label = reshaped_label.numpy()
        # reshaped_label = reshaped_label.astype(np.uint8)

        h, w = reshaped_preds_crf.shape
        # reshaped_label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        # for gray, rgb in label_to_color.items():
        #     reshaped_label_rgb[reshaped_label == gray, :] = rgb
        reshaped_preds_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for gray, rgb in label_to_color.items():
            reshaped_preds_crf_rgb[reshaped_preds_crf == gray, :] = rgb
        reshaped_preds_no_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for gray, rgb in label_to_color.items():
            reshaped_preds_no_crf_rgb[reshaped_preds_no_crf == gray, :] = rgb

        # Image.fromarray(reshaped_img).save(join(join(result_dir, "img", str(img_num) + ".png")))
        Image.fromarray(reshaped_preds_crf_rgb).save(
            join(join(result_dir, "cluster", str(img_num) + "_crf.png"))
        )
        Image.fromarray(reshaped_preds_no_crf_rgb).save(
            join(join(result_dir, "cluster", str(img_num) + "_no_crf.png"))
        )
        # Image.fromarray(reshaped_label_rgb).save(
        #     join(join(result_dir, "label", str(img_num) + ".png"))
        # )
        print("Saved image number ", img_num)


if __name__ == "__main__":
    prep_args()
    my_app()
