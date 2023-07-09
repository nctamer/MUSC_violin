from musc.model import FourHeads
from musc.representations import PerformanceLabel
import torch
from torch import nn
import numpy as np
import os
import json
import gdown

class PretrainedModel(FourHeads):
    def __init__(self, instrument='violin'):
        assert instrument in ['violin', 'Violin', 'vln', 'vl'], 'As of now, the only supported instrument is the violin'
        instrument = 'violin'
        package_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(package_dir, "musc", instrument + ".json"), "r") as f:
            args = json.load(f)
        labeling = PerformanceLabel(note_min=args['note_low'], note_max=args['note_high'],
                                    f0_bins_per_semitone=args['f0_bins_per_semitone'],
                                    f0_tolerance_c=200,
                                    f0_smooth_std_c=args['f0_smooth_std_c'], onset_smooth_std=args['onset_smooth_std'])

        super().__init__(pathway_multiscale=args['pathway_multiscale'],
                         num_pathway_layers=args['num_pathway_layers'], wiring=args['wiring'],
                         hop_length=args['hop_length'], chunk_size=args['chunk_size'],
                         labeling=labeling, sr=args['sampling_rate'])
        self.model_url = args['model_file']
        self.load_weight(instrument)
        self.eval()

    def load_weight(self, instrument):
        self.download_weights(instrument)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "{}_model.pt".format(instrument)
        self.load_state_dict(torch.load(os.path.join(package_dir, filename)))

    def download_weights(self, instrument):
        weight_file = "{}_model.pt".format(instrument)
        package_dir = os.path.dirname(os.path.realpath(__file__))
        weight_path = os.path.join(package_dir, weight_file)
        if not os.path.isfile(weight_path):
            package_dir = os.path.dirname(os.path.realpath(__file__))
            weight_path = os.path.join(package_dir, weight_file)
            if not os.path.exists(weight_path):
                gdown.download(f"https://drive.google.com/uc?export=download&confirm=pbef&id={self.model_url}", weight_path)
    
    @staticmethod
    def download_youtube(url, audio_codec='wav'):
        from yt_dlp import YoutubeDL
        ydl_opts = {'no-playlist': True, 'quiet': True, 'format': 'bestaudio/best',
                    'outtmpl': '%(id)s.%(ext)s', 'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_codec,
                'preferredquality': '192', }], }
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_id = info_dict.get('id', None)
            title = info_dict.get('title', None)
            ydl.download([url])
        return video_id + '.' + audio_codec, video_id, title

    def transcribe_youtube(self, url, audio_codec='wav', batch_size=64,
                           postprocessing='spotify', include_pitch_bends=True):
        file_path, video_id, title = self.download_youtube(url, audio_codec=audio_codec)
        midi = self.transcribe(file_path, batch_size=batch_size,
                               postprocessing=postprocessing, include_pitch_bends=include_pitch_bends)
        return midi, video_id, title  

