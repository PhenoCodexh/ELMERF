import os
import mmengine
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class DiceWeightLoggerHook(Hook):
    def __init__(self, log_every=100, out_dir=None):
        self.log_every = log_every
        self.out_dir = out_dir
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 获取 decode_head
        decode_head = runner.model.module.decode_head \
            if hasattr(runner.model, "module") else runner.model.decode_head

        if hasattr(decode_head, "aux_head"):
            aux_head = decode_head.aux_head
            crit = getattr(aux_head, "loss_decode", None)

            if crit is not None and hasattr(crit, "_last_w_pos"):
                w_pos = crit._last_w_pos
                w_neg = crit._last_w_neg
                iter = runner.iter

            if w_pos is not None and runner.iter % self.log_every == 0:
                mean_pos = w_pos.mean().item()
                mean_neg = w_neg.mean().item() if w_neg is not None else -1
                print(f"[DiceWeightLoggerHook] Iter {runner.iter} | Avg w_pos = {mean_pos:.4f} | Avg w_neg = {mean_neg:.4f}")

                # 保存权重
                if self.out_dir:
                    fname = os.path.join(self.out_dir, f"iter_{runner.iter}_w.txt")
                    with open(fname, "w") as f:
                        f.write(f"# Iter {runner.iter}\n")
                        f.write(f"# w_pos: {' '.join([f'{v:.4f}' for v in w_pos.view(-1).cpu().numpy()])}\n")
                        if w_neg is not None:
                            f.write(f"# w_neg: {' '.join([f'{v:.4f}' for v in w_neg.view(-1).cpu().numpy()])}\n")
