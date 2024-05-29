# Code taken from https://github.com/0nutation/SpeechGPT with some modifications
# Original code is under Apache License 2.0
import argparse
import os

import fairseq
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

# logging.basicConfig(
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=os.environ.get("LOGLEVEL", "INFO").upper(),
#     stream=sys.stdout,
# )
# logger = logging.getLogger("generate_pseudo_language")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureReader(object):
    def __init__(
        self,
        ckpt_path,
        layer,
        max_chunk=1600000,
        fp16=False,
        sampling_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().to(device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()

        self.layer_shift = 0
        self.target_sample_hz = sampling_rate

        logger.info(f"TASK CONFIG:\n{self.task.cfg}")

    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform):
        x = waveform
        with torch.no_grad():
            if self.fp16:
                x = x.half().cuda()
            else:
                x = x.float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer + self.layer_shift,
                )

                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self,
        ckpt_dir,
        layer=11,
        max_chunk=1600000,
        fp16=False,
        sampling_rate=16000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            ckpt_dir(str): path to hubert model dir(e.g. hubert_base_ls960.pt)
            layer(int): feat from which layer of hubert models defauly by 9
            max_chunk(int): default by 1600000
            fp16(bool): default by False
            sampling_rate(int): sampling_rate default by 16000
        """
        super().__init__()
        self.device = device

        ckpt_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3.pt")
        km_path = os.path.join(ckpt_dir, "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")

        self.feature_reader = FeatureReader(
            ckpt_path, layer, max_chunk, fp16, sampling_rate, device=device
        )
        self.apply_kmeans = ApplyKmeans(km_path)

    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i + 1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def __call__(self, path=None, waveform=None, merged=True):
        if waveform is None:
            waveform = self.feature_reader.read_audio(path)

        waveform = waveform.to(self.device)

        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        # merged_units = (
        #     "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        # )
        # unmerged_units = (
        #     "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"
        # )
        merged_units = dup_cluster_list
        unmerged_units = cluster_ids

        if merged:
            return merged_units
        else:
            return unmerged_units


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str)
    args = parser.parse_args()

    ckpt_dir = "speechgpt/utils/speech2unit/"

    s2u = Speech2Unit(ckpt_dir=ckpt_dir)

    units = s2u(args.wav)
    print(units)
