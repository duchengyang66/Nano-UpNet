"""Persistence of best validation metrics."""

import json
import os
import time


class BestMetrics:
    def __init__(self, save_dir, filename='best_metrics.json'):
        self.save_dir = save_dir
        self.path = os.path.join(save_dir, filename)
        self.best_psnr = -1e9
        self.best_epoch = -1
        self.updated_at = ''
        self._load()

    def _load(self):
        if os.path.isfile(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                self.best_psnr = float(data.get('best_psnr', -1e9))
                self.best_epoch = int(data.get('best_epoch', -1))
                self.updated_at = str(data.get('updated_at', ''))
            except Exception as e:
                print('BestMetrics: failed to load %s: %s' % (self.path, e))

    def update(self, psnr, epoch):
        if psnr > self.best_psnr:
            self.best_psnr = float(psnr)
            self.best_epoch = int(epoch)
            self.updated_at = time.strftime('%Y-%m-%dT%H:%M:%S')
            self.save()
            return True
        return False

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump({
                'best_psnr': self.best_psnr,
                'best_epoch': self.best_epoch,
                'updated_at': self.updated_at,
            }, f, indent=2)

    def __repr__(self):
        return 'BestMetrics(best_psnr=%.4f, best_epoch=%d)' % (self.best_psnr, self.best_epoch)
