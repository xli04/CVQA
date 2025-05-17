"""
Main training and evaluation script for Continual Visual Question Answering (CVQA).
Implements MM-Prompt model training with cross-modal prompt learning and memory replay.
"""

import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args
from vqa_data_memory import get_loader, get_loader_test, VQADataset, get_loader_memory
from utils import load_state_dict, LossMeter, set_global_logging_level
import dist_utils
import json
import random
from cam import generate_cam_for_test_set


proj_dir = Path(__file__).resolve().parent.parent

# Set up mixed precision training options
_use_native_amp = False
_use_apex = False

# Check PyTorch version to determine which AMP implementation to use
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from trainer_base import TrainerBase
# Import task definitions and evaluation utilities
from Question_type import All_task, Comp_task, show_results_matrix, evaluate_metric, Category_splits, ImgId_cate_map, QuesId_task_map, random_dic

# Global tracking variables for forgetting metrics
inner_valid_forg = 0
inner_topk_forg = 0
inner_train_num = 0

def cycle(iterable):
    """
    Create an infinite iterator that cycles through the items with shuffling.
    Used for memory replay in continual learning.
    """
    while True:
        for i in iterable:
            yield i

class Trainer(TrainerBase):
    """
    Trainer class for CVQA model with memory replay and task incremental learning.
    Handles training, validation, and testing across multiple tasks.
    """
    def __init__(self, args, coco_Ours, train_loader=None, val_loader=None, test_loader=None, train=True):
        """
        Initialize the trainer with task sequences and datasets.
        
        Args:
            args: Command-line arguments
            coco_Ours: List of tasks to train on
            train_loader: Training data loader (optional)
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
            train: Whether to initialize for training (True) or evaluation (False)
        """
        # Initialize results matrix for tracking performance across tasks
        self.result_matrix = {}
        self.task_list = []
        for task in coco_Ours:
            print(task)
            self.result_matrix[task] = {}
            self.task_list.append(task)

        # Create dictionaries to store data loaders for each task
        self.train_loader_dict = {}
        self.val_loader_dict = {}
        self.test_loader_dict = {}
        self.test_loader_dict_all = {}

        # Initialize datasets
        self.train_dset = VQADataset(args.train, True)
        self.val_dset = VQADataset(args.valid, True)
        self.test_dset = VQADataset(args.test, True)

        # Store Class Activation Maps
        self.stored_CAM = []
        
        # Initialize parent class
        super().__init__(
            args,
            train=train)

        # Suppress unnecessary logging
        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        # Import model class
        from vqa_model import VLT5VQA

        # Set up model based on backbone architecture
        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5VQA

        # Create model configuration and tokenizer
        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        
        # For BART tokenizer, add special tokens for visual features
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                        [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids([f'<vis_extra_id_{i}>' for i in range(100)])

        # Initialize the model
        self.model = self.create_model(model_class, config, **model_kwargs)

        # Resize token embeddings to match tokenizer vocabulary
        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.model.model.shared.num_embeddings + num_added_toks)

        self.model.tokenizer = self.tokenizer

        # Load checkpoint if specified
        self.start_epoch = None
        if args.load is not None:
            ckpt_path = args.load + '.pth'
            self.load_checkpoint(ckpt_path)

        # Initialize weights from scratch if needed
        if self.args.from_scratch:
            if args.ifseed:
                self.init_weights(seed=args.seed, ifseed=True)
            else:
                self.init_weights()

        # Move model to GPU
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Setup distributed training if needed
        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')
            
        # Store training configuration
        self.iftrain = train
        self.coco_Ours = coco_Ours

        # Track which tasks have been trained
        self.task_iftrain = {}
        for task in self.coco_Ours:
            self.task_iftrain[task] = 0

        # Track total examples per task
        self.task_total_num = torch.zeros(len(self.task_list))

        # Memory replay configuration
        self.M = args.m_size  # Memory size
        self.Examplar_set = {'G0': [], 'G1': [], 'G2': [], 'G3': [], 'G4': []}
        self.composition_test_cate = args.comp_cate

    def train(self, load=False):
        """
        Train the model on multiple tasks sequentially with memory replay.
        
        Args:
            load: Whether to load a checkpoint and continue training
        """
        if load == True:
            # Load checkpoint from previous task
            latest_task = args.checkpoint
            latest_task_idx = self.task_list.index(latest_task)
            for idx, task in enumerate(self.task_list):
                if idx <=latest_task_idx:
                    self.task_iftrain[task] = 1

            checkpoint_model = self.args.output + '/' + args.checkpoint+'_LAST'
            self.load(checkpoint_model)
            print('Success to load the checkpoint from the task ', latest_task)

        else:
            latest_task_idx = -1

        # Train on each task sequentially
        for task_idx, task in enumerate(self.task_list[latest_task_idx+1:]):
            print('======================== Now is task "', task, '" ========================')
            self.task_iftrain[task] = 1

            # Setup memory replay if enabled
            if args.memory:
                print("Use memory")
                if task_idx != latest_task_idx + 1:
                    print("load memory")
                    # Calculate memory allocation per task
                    each_memory = int(self.M / task_idx)
                    # Load data for previous task
                    data_info_path = ('/root/autodl-tmp/datasets/vqa/Partition_Q_reverse/karpathy_train_task_' + f'{self.task_list[task_idx - 1]}.json')
                    with open(data_info_path) as f:
                        data_info_dicts = json.load(f)

                    # Shuffle data for random sampling
                    random.shuffle(data_info_dicts)
                    # Allocate memory equally across categories
                    each_memory_for_cate = int(each_memory / 5)

                    # Sample examples for each category
                    for cate in Category_splits[str(task_idx)]:
                        num = 0
                        self.Examplar_set[cate].append([])
                        for _d in data_info_dicts:
                            question_id = str(_d['question_id'])
                            if question_id in QuesId_task_map:
                                if QuesId_task_map[question_id] in Category_splits[str(task_idx)][cate]:
                                    self.Examplar_set[cate][task_idx - 1].append(_d)
                                    num += 1
                                    if num >= each_memory_for_cate:
                                        break

                    print('Load from Partition_Q_v3......')

                    for cate in Category_splits[str(task_idx)]:
                        for i in range(task_idx):
                            self.Examplar_set[cate][i] = self.Examplar_set[cate][i][: each_memory_for_cate]

                    All_examplar = []
                    for E_set in self.Examplar_set:
                        for task_set in self.Examplar_set[E_set]:
                            All_examplar += task_set
                    # assert len(All_examplar) == M
                    print("# The size of the cate Memory:", len(All_examplar))
                else:
                    All_examplar = []
                    each_memory = 0
            else:
                All_examplar = []
                each_memory = 0

            # Load the data
            print("#Loading ", task)

            train_loader, total_num_Q = get_loader(
                args,
                str(task),
                self.coco_Ours,
                [],
                self.train_dset,
                split=args.train, mode='train', batch_size=args.batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=args.num_workers,
                topk=args.train_topk,
                task=task,
            )

            self.task_total_num[task_idx] = total_num_Q


            if args.valid_batch_size is not None:
                self.valid_batch_size = args.valid_batch_size
            else:
                self.valid_batch_size = args.batch_size
            print(f'Building val loader at GPU {args.gpu}')
            val_loader, _ = get_loader(
                args,
                str(task),
                self.coco_Ours,
                [],
                self.val_dset,
                split=args.valid, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )

            print(f'Building test loader at GPU {args.gpu}')
            test_loader, _ = get_loader(
                args,
                str(task),
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict[task] = test_loader

            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=self.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

            print("#Loading ", str(task))
            if len(All_examplar) > 0:
                memory_loader = get_loader_memory(
                    args,
                    str(task),
                    self.coco_Ours,
                    All_examplar,
                    self.train_dset,
                    split=args.train, mode='train', batch_size=args.batch_size,
                    distributed=args.distributed, gpu=args.gpu,
                    workers=args.num_workers,
                    topk=args.train_topk,
                )
            else:
                memory_loader = None

            if self.verbose:
                loss_meter = LossMeter()
                loss_meter_mem = LossMeter()
                # loss_mem_V = LossMeter()
                # loss_mem_Q_new = LossMeter()
                # loss_mem_V_new = LossMeter()
                best_valid = 0.
                best_epoch = 0

                if 't5' in self.args.backbone:
                    if self.args.use_vision:
                        project_name = "VLT5_VQA"
                    else:
                        project_name = "T5_VQA"
                elif 'bart' in self.args.backbone:
                    if self.args.use_vision:
                        project_name = "VLBart_VQA"
                    else:
                        project_name = "Bart_VQA"

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)
                # wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

            if self.args.distributed:
                dist.barrier()

            global_step = 0
            Category_splits_random = random_dic(Category_splits[str(task)])
            all_val_loader = {}
            last_result = {}
            pointer = 0
            visited = []
            for idx, cateGroup in enumerate(Category_splits_random):
                print('-------- Training the cate group ', cateGroup,' of task ', task,'------')
                pointer+=1
                visited.append(cateGroup)
                self.train_loader_cate = train_loader[cateGroup]
                self.val_loader_cate = val_loader[cateGroup]
                if memory_loader == None:
                    self.memory_loader_cate = []
                else:
                    self.memory_loader_cate = memory_loader[cateGroup]
                all_val_loader[cateGroup] = val_loader[cateGroup]
                last_result[cateGroup] = {}
                # Optimizer
                if self.iftrain:
                    if len(self.memory_loader_cate) > 0:
                        total_train_num = 2 * len(self.train_loader_cate.dataset)
                    else:
                        total_train_num = len(self.train_loader_cate.dataset)
                    self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler(total_train_num)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler = torch.cuda.amp.GradScaler()
                    elif _use_apex:
                        self.model, self.optim = amp.initialize(
                            self.model, self.optim, opt_level='O1', verbosity=self.verbose)

                if cateGroup == self.composition_test_cate and task != self.task_list[latest_task_idx+1]:
                    print("-------- Pass the training for", cateGroup, 'for after composition testing.--------')
                    continue


                for epoch in range(self.args.epochs):
                    print(f"Epoch {epoch + 1}/{self.args.epochs}")
                    if self.start_epoch is not None:
                        epoch += self.start_epoch
                    self.model.train()

                    if self.args.distributed:
                        self.train_loader_cate.sampler.set_epoch(epoch)
                    if self.verbose:
                        pbar = tqdm(total=len(self.train_loader_cate), ncols=120)

                    epoch_results = {
                        'loss': 0.,
                    }

                    quesid2ans = {}

                    if len(self.memory_loader_cate) > 0:
                        now_loader = zip(self.train_loader_cate, cycle(self.memory_loader_cate))
                        print('Use memory loader')
                    else:
                        now_loader = self.train_loader_cate

                    for now_batch in now_loader:
                        if len(now_batch) == 2:
                            batch, mem_batch = now_batch
                        else:
                            batch = now_batch
                            mem_batch = None

                        results, lr = self.train_step(batch, epoch_results, task_idx, each_memory)
                        if mem_batch:
                            results_mem, lr = self.train_step(mem_batch, epoch_results, task_idx, each_memory)

                        if self.verbose:
                            loss_meter.update(results['loss'].item())
                            desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                            desc_str += f' | Loss {loss_meter.val:4f}'
                            if mem_batch:
                                loss_meter_mem.update(results_mem['loss'].item())
                                desc_str += f' | Loss_mem {loss_meter_mem.val:4f}'
                            else:
                                loss_meter_mem.update(-1)


                            pbar.set_description(desc_str)
                            pbar.update(1)

                        if self.args.distributed:
                            dist.barrier()
                        # break

                    if self.verbose:
                        pbar.close()

                    print("Loss:",loss_meter.val,' Loss_mem:', loss_meter_mem.val)

                    # Validation
                    score_dict = self.evaluate(self.val_loader_cate)

                    if self.verbose:
                        valid_score = score_dict['topk_score'] * 100.
                        valid_score_raw = score_dict['overall']
                        last_result[cateGroup][cateGroup] = {}
                        last_result[cateGroup][cateGroup]['topk_score'] = valid_score
                        last_result[cateGroup][cateGroup]['overall'] = valid_score_raw

                        log_str = ''
                        log_str += "\nGroup %s Epoch %d: Valid Raw %0.2f Topk %0.2f" % (cateGroup, epoch, valid_score_raw, valid_score)

                        print(log_str)
                        if pointer > 1:
                            for prev_cate in visited:
                                if prev_cate == cateGroup:
                                    continue
                                last_result[cateGroup][prev_cate] = {}
                                score_prev = self.evaluate(all_val_loader[prev_cate])
                                prev_valid_score = score_prev['topk_score'] * 100.
                                prev_valid_score_raw = score_prev['overall']
                                before_result = last_result[prev_cate]
                                last_result[cateGroup][prev_cate]['topk_score'] = prev_valid_score
                                last_result[cateGroup][prev_cate]['overall'] = prev_valid_score_raw
                                best_raw = -1
                                best_topk = -1
                                for category in visited:
                                    if category == cateGroup:
                                        continue
                                    check_result = last_result[category]
                                    if not prev_cate in check_result:
                                        continue
                                    this_raw = last_result[category][prev_cate]['overall']
                                    this_topk = last_result[category][prev_cate]['topk_score']
                                    if this_raw > best_raw:
                                        best_raw = this_raw
                                    if this_topk > best_topk:
                                        best_topk = this_topk
                                print("Validation for " + prev_cate + " after training " + cateGroup + ": Valid Raw " + str(prev_valid_score_raw) + " Topk " + str(prev_valid_score))
                                gaps_raw = best_raw - prev_valid_score_raw
                                gaps = best_topk - prev_valid_score
                                print("Forget for Valid Raw: " + str(gaps_raw) + " Forget for Topk: " + str(gaps))
                                if(pointer == 2):
                                    global inner_valid_forg, inner_topk_forg, inner_train_num
                                    inner_valid_forg += gaps_raw
                                    inner_topk_forg += gaps
                                    inner_train_num += 1
                                

                    if self.args.distributed:
                        dist.barrier()

            if self.verbose:
                self.save(str(task) + "_LAST")

            # ========= Testing =========
            self.test(str(task))

        print("Saving the trained model and tokenizer...")
        self.model.save_pretrained(Path(''))
        self.tokenizer.save_pretrained(Path(''))
        print("Model and tokenizer saved successfully!")

    def train_step(self, batch, epoch_results, task_idx, each_memory):
        """
        Perform a single training step, with added functionality for managing GPU memory.
        """
        # Mixed precision training (if enabled)
        if self.args.fp16 and _use_native_amp:
            with autocast():
                if self.args.distributed:
                    results = self.model.train_step(batch)
                else:
                    results = self.model.train_step(batch)
        else:
            if self.args.distributed:
                results = self.model.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num)
            else:
                results = self.model.train_step(batch, task_idx, self.args.proto_alpha, self.args.proto_beta, each_memory, self.task_total_num)
    
        loss = results['loss']
        lambda_Q = self.args.lambda_Q
        lambda_V = self.args.lambda_V
        lambda_Q_new = self.args.lambda_Q_new
        lambda_V_new = self.args.lambda_V_new
    
        # Apply additional loss terms if available (memory loss, new memory loss)
        if 'loss_memory' in results:
            (loss_memory_Q, loss_memory_V) = results['loss_memory']
            loss = loss + lambda_Q * loss_memory_Q + lambda_V * loss_memory_V
        if 'loss_memory_new' in results:
            (loss_memory_Q_new, loss_memory_V_new) = results['loss_memory_new']
            loss = loss + lambda_Q_new * loss_memory_Q_new + lambda_V_new * loss_memory_V_new
    
        # Backward pass (with mixed precision if enabled)
        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    
        loss = loss.detach()

        self.model.update_prompt_fusion_attention()
    
        # Update Parameters
        if self.args.clip_grad_norm > 0:
            if self.args.fp16 and _use_native_amp:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)
            elif self.args.fp16 and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(
                    self.optim), self.args.clip_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.clip_grad_norm)
    
        if self.args.fp16 and _use_native_amp:
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.step()
    
        if self.lr_scheduler:
            self.lr_scheduler.step()
    
        # Zero gradients for the next step
        for param in self.model.parameters():
            param.grad = None
    
        # Update epoch results with any metrics from the step
        for k, v in results.items():
            if k in epoch_results:
                epoch_results[k] += v.item()
    
        # Learning rate
        if self.lr_scheduler:
            if version.parse(torch.__version__) >= version.parse("1.4"):
                lr = self.lr_scheduler.get_last_lr()[0]
            else:
                lr = self.lr_scheduler.get_lr()[0]
        else:
            try:
                lr = self.optim.get_lr()[0]
            except AttributeError:
                lr = self.args.lr
    
        return results, lr


    def Test(self, load=False):

        for task_idx, task in enumerate(self.task_list):
            print('======================== Now is task "', str(task), '" ========================')

            test_loader = get_loader_test(
                args,
                self.coco_Ours,
                [],
                self.test_dset,
                split=args.test, mode='val', batch_size=args.valid_batch_size,
                distributed=args.distributed, gpu=args.gpu,
                workers=4,
                topk=args.valid_topk,
                task=task,
            )
            self.test_loader_dict_all[task] = test_loader

        # ========= Testing =========
        self.test(self.task_list[-1])


    def test(self, task, comp=False):
        # Test Set
        if not os.path.exists(self.args.output):
            os.mkdir(self.args.output)
        last_path = os.path.join(self.args.output, str(task) + '_LAST')
        self.load(last_path)

        # =========== test for all previous tasks
        flag = 1
        for test_task in self.coco_Ours:
            if self.args.now_train:
                if self.task_iftrain[test_task] == 0:
                    flag = 0
            if flag == 1:
                self.test_loader = self.test_loader_dict_all[test_task]
                print(' ===== Test for the task "' + str(test_task) + '"  ======')

                quesid2ans = self.predict(self.test_loader, task, test_task)

                if self.verbose:
                    evaluator = self.test_loader.evaluator
                    score_dict = evaluator.evaluate(quesid2ans)

                    acc_dict_all = evaluator.evaluate_raw(quesid2ans)
                    acc_dict_answerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=True)
                    acc_dict_unanswerable = evaluator.evaluate_raw(quesid2ans, is_topk_optimal=False)

                    wandb_log_dict = {}
                    wandb_log_dict['Test/overall'] = acc_dict_all['overall']
                    wandb_log_dict['Test/topk_optimal'] = acc_dict_answerable['overall']
                    wandb_log_dict['Test/topk_not_optimal'] = acc_dict_unanswerable['overall']

                    # self.print_accuracy_per_class_and_question_type(quesid2ans)

                    for qtype, score in acc_dict_all['perQuestionType'].items():
                        wandb_log_dict[f'Test_Qtypes/{qtype}'] = score
                    for atype, score in acc_dict_all['perAnswerType'].items():
                        if atype == 'yes/no':
                            atype = 'yes_no'
                        wandb_log_dict[f'Test_Atypes/{atype}'] = score

                    print(test_task, wandb_log_dict)
                self.result_matrix[int(task)][test_task] = acc_dict_all['overall']

                if self.args.distributed:
                    dist.barrier()
                    
    def print_accuracy_per_class_and_question_type(self, quesid2ans):
        # Create a dictionary for fast lookup of question_id
        dataset_dict = {datum['question_id']: datum for datum in self.test_loader.dataset}
    
        # Initialize accuracy tracking
        class_accuracies = {i: {'total': 0, 'correct': 0, 'question_types': {}} for i in range(1, 91)}
    
        # Loop through all question ids and answers
        for qid, ans in quesid2ans.items():
            # Access the datum directly using question_id lookup
            datum = dataset_dict.get(qid)
            if not datum:
                continue  # Skip if the datum for the current qid is not found
    
            img_id = datum['img_id']  # Image ID (to get category)
            question_id = datum['question_id']  # Question ID (to get question type)
    
            # Map img_id to img_cate using ImgId_cate_map
            img_cate = ImgId_cate_map.get(img_id, None)
            if img_cate is None:
                continue  # Skip if no category is found for img_id
    
            # Map question_id to question_type using QuesId_task_map
            question_type = QuesId_task_map.get(str(question_id), None)
            if question_type is None:
                continue  # Skip if no question type is found for question_id
    
            # Get ground truth answers
            gt_answers = [key for key, _ in datum['label'].items()]  # Extract ground truth answers from the 'label' field
    
            # Determine if the prediction is correct (Exact Match)
            correct = 1 if ans in gt_answers else 0
    
            # Track overall accuracy per class
            class_accuracies[img_cate]['total'] += 1
            class_accuracies[img_cate]['correct'] += correct
    
            # Track accuracy for each question type within the class
            if question_type not in class_accuracies[img_cate]['question_types']:
                class_accuracies[img_cate]['question_types'][question_type] = {'correct': 0, 'total': 0}
    
            class_accuracies[img_cate]['question_types'][question_type]['total'] += 1
            class_accuracies[img_cate]['question_types'][question_type]['correct'] += correct
    
        # After processing all predictions, print the per-class and per-question-type accuracy
        for cate, data in class_accuracies.items():
            print(f"Class {cate}:")
            overall_accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
            print(f"  - Overall Accuracy: {overall_accuracy:.2f}%")
    
            # Print per question type accuracy for this class
            for qtype, stats in data['question_types'].items():
                question_accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
                print(f"    - {qtype} Accuracy: {question_accuracy:.2f}%")



    def predict(self, loader, task, test_task, dump_path=None, test=True):
        self.model.eval()
        dataset_dict = {datum['question_id']: datum for datum in loader.dataset}
        with torch.no_grad():
            # answer_time = {"yes": {}, "no": {}, "0": {}, "1": {}}
            answer_time = {}
            answer_time_que = {}
            # answer_time = {, "no": {}}
            quesid2ans = {}
            question_to_attention = {}
            all_input_ids = {}
            all_ques_ids = []
            category = {}
            # yes_no = {"total_yes": 0, "total_no": 0, "correct_yes": 0, "correct_no": 0}
            # count = {"total_0": 0, "total_1": 0, "correct_0": 0, "correct_1": 0, "total": 0}
            if self.verbose:
                pbar = tqdm(total=len(loader), ncols=120, desc="Prediction---")
            for i, batch in enumerate(loader):
                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)
                pred_ans = results['pred_ans'] # generated_sents
                ques_ids = batch['question_ids']
                # heatmap
                # if test == True:
                #     # add all the related information in to the map under the key qid
                #     for qid, inp_id in zip(ques_ids, results['input_ids']):
                #         if not qid in self.stored_CAM:
                #             continue
                #         category[qid] = {}
                #         all_input_ids[qid] = inp_id
                #         all_ques_ids.append(qid)
                #     for layer_idx, layer_attn in enumerate(results['cross_attention']):
                #         for question_idx, question_id in enumerate(ques_ids):
                #             if not question_id in self.stored_CAM:
                #                 continue
                #             if question_id not in question_to_attention:
                #                 question_to_attention[question_id] = []
                #             question_to_attention[question_id].append(layer_attn[question_idx])  # [num_heads, seq_len, num_regions]
                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans 
                    # heatmap
                    # if test == True:
                    #     data_dic = dataset_dict.get(qid)
                    #     img_cate = ImgId_cate_map.get(data_dic['img_id'], None)
                    #     if img_cate not in answer_time:
                    #         answer_time[img_cate] = {}
                    #     time = answer_time[img_cate]
                    #     if ans not in time:
                    #         time[ans] = 0
                    #     current = time[ans] + 1
                    #     time[ans] = current
                    #     answer_time[img_cate] = time

                if self.verbose:
                    pbar.update(1)
            # heatmap
            # if test == True:
            #     # stack the attention
            #     attention_imgId_boxes_answer = {}
            #     for question_id, layer_attentions in question_to_attention.items():
            #         # Stack the layer attention maps: [num_layers, num_heads, seq_len, num_regions]
            #         stacked_layers = torch.stack(layer_attentions, dim=0)
            #         datum = dataset_dict.get(question_id)
            #         attention_imgId_boxes_answer[question_id] = {
            #             "attentions": stacked_layers,
            #             "img_id": datum['img_id'],
            #             "boxes": datum['boxes'],
            #             "answer": quesid2ans[question_id],
            #             "inp_id": all_input_ids[question_id]
            #         }
            #         img_cate = ImgId_cate_map.get(datum['img_id'], None)
            #         que_cate = QuesId_task_map.get(str(question_id), None)
            #         print(que_cate)
            #         print(question_id)
            #         category[question_id] = {}
            #         category[question_id]["img_cate"] = img_cate
            #         category[question_id]["que_cate"] = que_cate
            #     print("aatention :", len(attention_imgId_boxes_answer))
            #     generate the heatmap
            
            #     generate_cam_for_test_set(self.tokenizer, all_ques_ids, attention_imgId_boxes_answer, task, test_task, category)


            if self.verbose:
                pbar.close()

        if self.args.distributed:
            dist.barrier()

        qid2ans_list = dist_utils.all_gather(quesid2ans)
        if self.verbose:
            quesid2ans = {}
            for qid2ans in qid2ans_list:
                for k, v in qid2ans.items():
                    quesid2ans[k] = v

            if dump_path is not None:
                evaluator = loader.evaluator
                evaluator.dump_result(quesid2ans, dump_path)

        return quesid2ans

    def evaluate(self, loader, dump_path=None):
        quesid2ans = self.predict(loader, 0, 0, dump_path, False)

        if self.verbose:
            evaluator = loader.evaluator
            acc_dict = evaluator.evaluate_raw(quesid2ans)
            topk_score = evaluator.evaluate(quesid2ans)
            acc_dict['topk_score'] = topk_score

            return acc_dict

def main_worker(gpu, args):
    # GPU assignment
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        print("Initializing distributed training...")
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
    else:
        print("Running on single GPU...")
        torch.cuda.set_device(args.gpu)

    print(f'Building train loader at GPU {gpu}')

    # Replace `coco_Ours` initialization with your actual dataset or task logic
    coco_Ours = All_task  # Dummy variable; replace as necessary

    try:
        print("Initializing Trainer...")
        trainer = Trainer(args, coco_Ours, train=True)
    except Exception as e:
        print(f"Error during Trainer initialization: {e}")
        raise

    if args.now_train:
        if args.checkpoint != 'None':
            print("Starting training process...")
            trainer.train(load=True)
        else:
            print("Starting training process without checkpoint...")
            trainer.train(load=False)

        print('#------------------ result_matrix --------------------#')
        show_results_matrix(trainer.result_matrix)
        path = args.output + 'results_matrix.json'
        # Uncomment this line to save the results matrix
        # save_results_matrix(trainer.result_matrix, path)
        metric_dict = evaluate_metric(trainer.result_matrix)
        print('#------  Metric  ------#')
        print('Incremental avg accuracy:', metric_dict['Incre_avg_acc'])
        print('*** Avg accuracy ***', metric_dict['Avg_acc'])
        print('Incremental avg forget:', metric_dict['Incre_avg_forget'])
        print('*** Avg forget ***', metric_dict['Avg_forget'])
        print('6Q Incremental avg accuracy:', metric_dict['Incre_avg_acc_6Q'])
        print('*** _6Q Avg accuracy ***', metric_dict['Avg_acc_6Q'])
        print('_6Q Incremental avg forget:', metric_dict['Incre_avg_forget_6Q'])
        print('*** _6Q Avg forget ***', metric_dict['Avg_forget_6Q'])
        print("Avg Inner Forgetting score_raw: " + str(round(inner_valid_forg/inner_train_num, 3)))
        print("Avg Inner Forgetting topk_score: " + str(round(inner_topk_forg/inner_train_num, 3)))
        # print out the answer times
    else:
        if args.checkpoint != 'None':
            print("Starting testing process with checkpoint...")
            trainer.Test(load=True)
        else:
            print("Starting testing process without checkpoint...")
            trainer.Test(load=False)

        try:
            print('#------------------ Final Performance --------------------#')
            print(trainer.result_matrix['q_causal'])
            acc = 0
            for key in trainer.result_matrix['q_causal']:
                acc += trainer.result_matrix['q_causal'][key]
            print('AP:', round(acc / 10, 4))
        except Exception as e:
            print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    print("Main block started...")
    cudnn.benchmark = True
    args = parse_args()
    args = parse_args()
    ngpus_per_node = 1
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)

        else:
            ckpt_str = 'scrach'
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name
        
        print("Starting main process...")
        main_worker(0, args)
