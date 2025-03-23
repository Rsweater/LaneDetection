
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class VLaneDetector(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        lane_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        """Unified style detector."""
        super(VLaneDetector, self).__init__(
            backbone=backbone, 
            neck=neck, 
            bbox_head=lane_head,
            train_cfg=train_cfg, 
            test_cfg=test_cfg, 
            pretrained=pretrained, 
            init_cfg=init_cfg
        )
        
    # Inherited from SingleStageDetector, overriding the forward_train and forward_test methods.
    # 
    # @auto_fp16(apply_to=('img', ))
    # def forward(self, img, img_metas, return_loss=True, **kwargs):
    #     if torch.onnx.is_in_onnx_export():
    #         assert len(img_metas) == 1
    #         return self.onnx_export(img[0], img_metas[0])

    #     if return_loss:
    #         return self.forward_train(img, img_metas, **kwargs)
    #     else:
    #         return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img_metas["imgs"], img_metas)
        x = self.extract_feat(img_metas["imgs"])
        losses = self.bbox_head.forward_train(x, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        assert (
            img.shape[0] == 1 and len(img_metas) == 1
        ), "Only single-image test is supported."
        img_metas[0]["batch_input_shape"] = tuple(img.size()[-2:])

        # For dynamic input image setting (e.g. CurveLanes、VIL-100 dataset)
        if "crop_offset" in img_metas[0]:
            self.bbox_head.test_cfg.cut_height = img_metas[0]["crop_offset"][1]
            self.bbox_head.test_cfg.ori_img_h = img_metas[0]["ori_shape"][0]
            self.bbox_head.test_cfg.ori_img_w = img_metas[0]["ori_shape"][1]

        x = self.extract_feat(img_metas["imgs"])
        output = self.bbox_head.simple_test(x)
        result_dict = {
            "result": output,
            "meta": img_metas[0],
        }
        return [result_dict]  # assuming batch size is 1
