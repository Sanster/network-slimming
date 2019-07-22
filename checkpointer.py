import os
import json
from collections import OrderedDict, namedtuple

import torch


class Checkpointer:
    MODEL_STATE = 'model_state_dict'
    OPTIMIZER_STATE = 'optimizer_state_dict'
    SCHEDULER_STATE = 'scheduler_state_dict'

    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir=None,
            max_count=5
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.max_count = max_count

        self.cache_file = None

        if save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if save_dir and os.path.exists(self.save_dir):
            self.cache_file = os.path.join(self.save_dir, 'cache_file.json')
            """
            {
                "last": "last_checkpoint_file_name.pth.tar",
                "checkpoints": [
                    "last_checkpoint_file_name1.pth.tar",
                    "last_checkpoint_filename2.pth.tar",
                    "last_checkpoint_file_name3.pth.tar",
                ]
            }
            checkpoints 按照数组里面的顺序，从旧到新
            """
            self.cache_data = self._load_cache_data()

    def save(self, name):
        if not self.save_dir:
            return

        data = {self.MODEL_STATE: self.model.state_dict()}
        if self.optimizer is not None:
            data[self.OPTIMIZER_STATE] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data[self.SCHEDULER_STATE] = self.scheduler.state_dict()

        model_save_path = os.path.join(self.save_dir, "{}.pth.tar".format(name))
        print("Saving checkpoint to {}".format(model_save_path))
        torch.save(data, model_save_path)

        record = {'path': model_save_path}
        self.cache_data['last'] = model_save_path
        self.cache_data['checkpoints'].insert(0, record)
        if len(self.cache_data['checkpoints']) > self.max_count:
            checkpoints_to_remove = self.cache_data['checkpoints'][self.max_count:]
            for it in checkpoints_to_remove:
                os.remove(it['path'])
            self.cache_data['checkpoints'] = self.cache_data['checkpoints'][:self.max_count]

        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_data, f, indent=2, ensure_ascii=False)

    def load(self, f=None):
        if f is None and self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.cache_data['last']

        if not f:
            # no checkpoint could be found
            raise FileNotFoundError("No checkpoint found. Initializing model from scratch")

        print("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

        if self.OPTIMIZER_STATE in checkpoint and self.optimizer:
            print("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop(self.OPTIMIZER_STATE))

        if self.SCHEDULER_STATE in checkpoint and self.scheduler:
            print("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop(self.SCHEDULER_STATE))

        # return any further checkpoint data
        return f

    def has_checkpoint(self):
        if not self.cache_file:
            return False

        return os.path.exists(self.cache_file)

    def _load_cache_data(self):
        if self.has_checkpoint():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        else:
            cache_data = {
                'last': '',
                'checkpoints': []
            }
        return cache_data

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        loaded_state_dict = checkpoint.pop(self.MODEL_STATE)
        model_state_dict = self.model.state_dict()
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        loaded_state_dict = self._strip_prefix_if_present(loaded_state_dict, prefix="module.")
        # align_and_update_state_dicts(model_state_dict, loaded_state_dict)

        # use strict loading
        self.model.load_state_dict(loaded_state_dict)

    def _strip_prefix_if_present(self, state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return state_dict
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict


class BestCheckpointer(Checkpointer):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir=None,
            max_count=5,
            best_value=None
    ):
        super(BestCheckpointer, self).__init__(model, optimizer, scheduler, save_dir, max_count)
        self.best_value = best_value

    def load(self):
        super(BestCheckpointer, self).load()
        if self.has_checkpoint():
            self.best_value = self.cache_data['best_value']

    def save(self, name):
        self.cache_data['best_value'] = self.best_value
        super(BestCheckpointer, self).save(name)
