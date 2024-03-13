from ginns_patches.data import image2PatchesOverlap, patches2ImageOverlap
import torch.multiprocessing
import hydra
from omegaconf import DictConfig
import os
import numpy as np
from PIL import Image
from train_segmentation import LitUnsupervisedSegmenter


def save_sample_patchwise(args, model, sample, filenamePrefix, tile_name):
    image1, image2, target = sample["image1"], sample["image2"], sample["label"]
    image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
    with torch.no_grad():
        image1_np = np.array(image1.detach().cpu()).squeeze()
        # so, hier wird das Bild zerstückelt. In diesem Fall entstehen patches von 1100x1100 pixel, die mit 100 Pixel überlappen.
        # Ganz sicher bin ich mir aber nicht mehr. Mach print(patches1[1,:,:,:].shape) oder so, damit du sicher gehst dass die patches 320x320 sind.
        patches1, metaInfo = image2PatchesOverlap(
            image1_np, 280, 280, 80
        )  # default: 1000,1000,100, overlap 100px
        output = patches1.astype(np.uint8)
        # for class probability
        for i in range(patches1.shape[0]):  # über alle patches iterieren.
            print(str(i))
            currentPatch1_np = patches1[i, :, :, :]
            currentPatch1_np = np.transpose(currentPatch1_np, axes=[2, 0, 1])
            currentPatch1 = torch.tensor(
                currentPatch1_np
            )  # von numpy array zu torch tensor
            currentPatch1 = torch.unsqueeze(currentPatch1, 0)
            outputPatch = model(
                currentPatch1, currentPatch1
            )  # hier wird es durch das Model gejagt.
            output[i, :, :, :] = outputPatch
        predictionImage = patches2ImageOverlap(
            output, metaInfo
        )  # und hier wieder alles zusammengefügt.

        # save image as tif
        im = Image.fromarray(predictionImage)
        filename = filenamePrefix + tile_name + "_prediction.tif"
        im.save(filename)


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

    # read orthophoto. Das alles musst du in einer Schleife machen, damit du alle 42 Ahrtal Bilder durchnudelst.
    ortho = Image.open(
        "O:/Studenten/jon86439/Ahrtal Road Detection/Images/uint8/RGB_val/uint8_00128_003927419_400.tif"
    )
    ortho = ortho.convert("RGB")

    # Modell initiieren blabla, wie in plot_potsdam
    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        "../saved_models/potsdam_test.ckpt"
    )

    filenamePrefix = "O:/Studenten/jon86439/Ahrtal Road Detection/test_output/"
    tile_name = "uint8_00128_003927419_400"

    save_sample_patchwise(model, sample, filenamePrefix, tile_name)

    # # saving every image:
    # for i in range(38):
    #     img_num = i
    #     output = {
    #         k: torch.cat(v, dim=0)[15 * 15 * img_num : 15 * 15 * (img_num + 1)]
    #         for k, v in outputs.items()
    #     }

    #     full_image = (
    #         output["img"]
    #         .reshape(15, 15, 3, 320, 320)
    #         .permute(2, 0, 3, 1, 4)
    #         .reshape(3, 320 * 15, 320 * 15)
    #     )

    #     # 3 cluster channels:
    #     # full_cluster_prob = output['cluster_prob'].reshape(15, 15, 3, 320, 320).permute(2, 0, 3, 1, 4).reshape(3, 320 * 15, 320 * 15)

    #     # 6 cluster channels:
    #     full_cluster_prob = (
    #         output["cluster_prob"]
    #         .reshape(15, 15, 6, 320, 320)
    #         .permute(2, 0, 3, 1, 4)
    #         .reshape(6, 320 * 15, 320 * 15)
    #     )

    #     crf_probs = dense_crf(
    #         full_image.cpu().detach(), full_cluster_prob.cpu().detach()
    #     )
    #     no_crf_probs = full_cluster_prob.numpy()

    #     reshaped_label = (
    #         output["label"]
    #         .reshape(15, 15, 320, 320)
    #         .permute(0, 2, 1, 3)
    #         .reshape(320 * 15, 320 * 15)
    #     )
    #     # reshaped_img = unnorm(full_image).permute(1, 2, 0)
    #     reshaped_preds_crf = model.test_cluster_metrics.map_clusters(
    #         np.expand_dims(crf_probs.argmax(0), 0)
    #     )
    #     reshaped_preds_no_crf = model.test_cluster_metrics.map_clusters(
    #         np.expand_dims(no_crf_probs.argmax(0), 0)
    #     )

    #     # reshaped_img = reshaped_img.numpy() * 255
    #     # reshaped_img = reshaped_img.astype(np.uint8)
    #     reshaped_preds_crf = reshaped_preds_crf.numpy()
    #     reshaped_preds_crf = reshaped_preds_crf.astype(np.uint8)
    #     reshaped_preds_no_crf = reshaped_preds_no_crf.numpy()
    #     reshaped_preds_no_crf = reshaped_preds_no_crf.astype(np.uint8)
    #     reshaped_label = reshaped_label.numpy()
    #     reshaped_label = reshaped_label.astype(np.uint8)

    #     h, w = reshaped_preds_crf.shape
    #     reshaped_label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     for gray, rgb in label_to_color.items():
    #         reshaped_label_rgb[reshaped_label == gray, :] = rgb
    #     reshaped_preds_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     for gray, rgb in label_to_color.items():
    #         reshaped_preds_crf_rgb[reshaped_preds_crf == gray, :] = rgb
    #     reshaped_preds_no_crf_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     for gray, rgb in label_to_color.items():
    #         reshaped_preds_no_crf_rgb[reshaped_preds_no_crf == gray, :] = rgb

    #     # Image.fromarray(reshaped_img).save(join(join(result_dir, "img", str(img_num) + ".png")))
    #     Image.fromarray(reshaped_preds_crf_rgb).save(
    #         join(join(result_dir, "cluster", str(img_num) + "_crf.png"))
    #     )
    #     Image.fromarray(reshaped_preds_no_crf_rgb).save(
    #         join(join(result_dir, "cluster", str(img_num) + "_no_crf.png"))
    #     )
    #     Image.fromarray(reshaped_label_rgb).save(
    #         join(join(result_dir, "label", str(img_num) + ".png"))
    #     )
    #     print("Saved image number ", img_num)


if __name__ == "__main__":
    prep_args()
    my_app()
