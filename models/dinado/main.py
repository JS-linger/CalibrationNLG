import argparse
import torch
import tqdm
import datasets
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from modeling_nado_generic import DiNADOMergeLMHeadModel
import numpy as np


class UnconditionalCommonGen(Dataset):
    def __init__(self, tokenizer, max_len=256):
        self.common_gen = datasets.load_dataset("common_gen")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len
        self.split = "train"

    def __len__(self):
        return len(self.common_gen[self.split])

    def eval(self):
        self.split = "validation"

    def train(self):
        self.split = "train"

    def __getitem__(self, item):
        target = self.common_gen[self.split][item]["target"]
        tokenized = self.tokenizer(
            f"{self.tokenizer.eos_token} " + target.strip() + self.tokenizer.eos_token,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len
        )
        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }

class ConditionalCommonGen(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, max_len=256):
        self.common_gen = datasets.load_dataset("common_gen")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len
        self.split = "train"

    def __len__(self):
        return len(self.common_gen[self.split])

    def eval(self):
        self.split = "validation"

    def train(self):
        self.split = "train"

    def __getitem__(self, item):
        concepts = self.common_gen[self.split][item]["concepts"]
        target = self.common_gen[self.split][item]["target"]
        tokenized_prefix = self.tokenizer(
            "<|endoftext|> " + " ".join(concepts) + " =", return_tensors="pt", padding="max_length", max_length=self.max_len
        )
        tokenized = self.tokenizer(
            "<|endoftext|> " + " ".join(concepts) + " = " + target.strip() + "<|endoftext|>", return_tensors="pt", padding="max_length", max_length=self.max_len
        )
        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "label_mask": (1 - tokenized_prefix.attention_mask[0]) * tokenized.attention_mask[0],
        }

class LabeledCommonGen(Dataset):
    def __init__(self, tokenizer: GPT2Tokenizer, dataset, max_len=384):
        self.common_gen = dataset
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_len = max_len

    def __len__(self):
        return len(self.common_gen)


    def __getitem__(self, item):
        concepts = self.common_gen[item]["concepts"]
        target = self.common_gen[item]["target"]
        tokenized_prefix = self.tokenizer(
            "<|endoftext|> " + concepts + " =", return_tensors="pt", padding="max_length", max_length=self.max_len
        )
        tokenized = self.tokenizer(
            "<|endoftext|> " + concepts + " = " + target.strip() + "<|endoftext|>", return_tensors="pt",
            padding="max_length", max_length=self.max_len
        )
        return {
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "label_mask": (1 - tokenized_prefix.attention_mask[0]) * (tokenized.input_ids[0] != self.tokenizer.pad_token_id).to(torch.long),
            "labels": self.common_gen[item]["label"]
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="alignment",
        type=str,
        required=False,
        help="Determine the stage of the pipeline.",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        required=False,
        help="Continue the training or start from scratch.",
    )
    parser.add_argument(
        "--dinado",
        default="merge",
        type=str,
        required=False,
        help="Determine the variant of the nado algorithm.",

    )
    parser.add_argument(
        "--grad_step",
        default=16,
        type=int,
        required=False,
        help="Continue the training or start from scratch.",
    )
    args = parser.parse_args()
    if args.stage == "pretrain":
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        model.cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

        dataset = UnconditionalCommonGen(tokenizer, 128)
        dataset[0]
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size // args.grad_step, shuffle=True, drop_last=True, pin_memory=True,
            num_workers=8
        )

        opt = AdamW(lr=5e-6, params=model.parameters(), betas=(0.9, 0.95))

        for epoch_idx in range(0):
            iter_idx = 0
            moving_avg = None
            dataset.train()
            model.train()
            iterator = tqdm.tqdm(dataloader, dynamic_ncols=True)
            for batch in iterator:
                input_ids = batch['input_ids']
                mask = batch['attention_mask']
                efft_length = mask.sum(dim=-1)
                batch_maxlen = efft_length.amax().item()
                input_ids = input_ids[:, 0:batch_maxlen].cuda()
                labels = input_ids[:, 1:].contiguous()
                mask = mask[:, 1:batch_maxlen].cuda()
                with torch.autocast(device_type='cuda'):
                    model_output = model(
                        input_ids=input_ids,
                    ).logits[:, :-1, :].log_softmax(dim=-1)

                nll = model_output.gather(
                    dim=-1, index=labels.unsqueeze(dim=-1)
                ).reshape_as(labels)
                nll = nll * mask

                if iter_idx % args.grad_step == 0:
                    opt.zero_grad()

                nll = -nll.sum(dim=-1).mean()

                if moving_avg is None:
                    moving_avg = nll.detach().cpu().item()
                else:
                    moving_avg = 0.95 * moving_avg + 0.05 * nll.detach().cpu().item()
                nll.backward()

                iter_idx += 1
                if iter_idx % args.grad_step == 0:
                    opt.step()
                    if (iter_idx // args.grad_step) % 10 == 0:
                        iterator.write("Pretrain Epoch %d-%d: Train NLL %f" % (epoch_idx, iter_idx, moving_avg))

            dataset.eval()
            model.eval()
            iterator = tqdm.tqdm(dataloader, dynamic_ncols=True)
            moving_avg = []
            for batch in iterator:
                input_ids = batch['input_ids']
                mask = batch['attention_mask']
                efft_length = mask.sum(dim=-1)
                batch_maxlen = efft_length.amax().item()
                input_ids = input_ids[:, 0:batch_maxlen].cuda()
                labels = input_ids[:, 1:].contiguous()
                mask = mask[:, 1:batch_maxlen].cuda()
                with torch.no_grad():
                    with torch.autocast(device_type='cuda'):
                        model_output = model(
                            input_ids=input_ids,
                        ).logits[:, :-1, :].log_softmax(dim=-1)

                    nll = model_output.gather(
                        dim=-1, index=labels.unsqueeze(dim=-1)
                    ).reshape_as(labels)
                    nll = nll * mask

                    nll = -nll.sum(dim=-1).mean()

                    moving_avg.append(nll.detach().cpu().item())

            print("Pretrain Epoch %d-%d: Dev NLL %s" % (epoch_idx, iter_idx, np.mean(moving_avg)))

            model.save_pretrained("gpt2_sft_DA_commongen")
        opt = AdamW(lr=1e-6, params=model.parameters(), betas=(0.9, 0.95))
        dataset = ConditionalCommonGen(tokenizer, 128)
        dataset[0]
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size // args.grad_step, shuffle=True, drop_last=True, pin_memory=True,
            num_workers=8
        )

        for epoch_idx in range(5):
            iter_idx = 0
            moving_avg = None
            dataset.train()
            model.train()
            iterator = tqdm.tqdm(dataloader, dynamic_ncols=True)
            for batch in iterator:
                input_ids = batch['input_ids']
                mask = batch['attention_mask']
                efft_length = mask.sum(dim=-1)
                batch_maxlen = efft_length.amax().item()
                mask = batch['label_mask']
                input_ids = input_ids[:, 0:batch_maxlen].cuda()
                labels = input_ids[:, 1:].contiguous()
                mask = mask[:, 1:batch_maxlen].cuda()
                with torch.autocast(device_type='cuda'):
                    model_output = model(
                        input_ids=input_ids,
                    ).logits[:, :-1, :].log_softmax(dim=-1)

                nll = model_output.gather(
                    dim=-1, index=labels.unsqueeze(dim=-1)
                ).reshape_as(labels)
                nll = nll * mask

                if iter_idx % args.grad_step == 0:
                    opt.zero_grad()

                nll = -nll.sum(dim=-1).mean()

                if moving_avg is None:
                    moving_avg = nll.detach().cpu().item()
                else:
                    moving_avg = 0.95 * moving_avg + 0.05 * nll.detach().cpu().item()
                (nll / args.grad_step).backward()

                iter_idx += 1
                if iter_idx % args.grad_step == 0:
                    opt.step()
                    if (iter_idx // args.grad_step) % 10 == 0:
                        iterator.write("Pretrain Epoch %d-%d: Train NLL %f" % (epoch_idx, iter_idx, moving_avg))

            dataset.eval()
            model.eval()
            iterator = tqdm.tqdm(dataloader, dynamic_ncols=True)
            moving_avg = []
            for batch in iterator:
                input_ids = batch['input_ids']
                mask = batch['attention_mask']
                efft_length = mask.sum(dim=-1)
                batch_maxlen = efft_length.amax().item()
                mask = batch['label_mask']
                input_ids = input_ids[:, 0:batch_maxlen].cuda()
                labels = input_ids[:, 1:].contiguous()
                mask = mask[:, 1:batch_maxlen].cuda()
                with torch.no_grad():
                    with torch.autocast(device_type='cuda'):
                        model_output = model(
                            input_ids=input_ids,
                        ).logits[:, :-1, :].log_softmax(dim=-1)

                    nll = model_output.gather(
                        dim=-1, index=labels.unsqueeze(dim=-1)
                    ).reshape_as(labels)
                    nll = nll * mask

                    nll = -nll.sum(dim=-1).mean()

                    moving_avg.append(nll.detach().cpu().item())

            print("Pretrain Epoch %d-%d: Dev NLL %s" % (epoch_idx, iter_idx, np.mean(moving_avg)))
        model.save_pretrained("gpt2_sft_commongen")
    elif args.stage == "pretrain_eval":
        model = GPT2LMHeadModel.from_pretrained("./gpt2_sft_commongen")
        model.cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        with open("evaluation/common_gen-keys-dev.txt", "r") as fin:
            with open("evaluation/tmp-generated-dev.txt", "w") as fout:
                iterator = tqdm.tqdm(fin.readlines(), dynamic_ncols=True)
                for line in iterator:
                    keys = "<|endoftext|> " + line.strip() + " ="
                    input_ids = tokenizer(
                        keys, return_tensors="pt",
                    )
                    with torch.autocast(device_type='cuda'):
                        generated = model.generate(
                            input_ids=input_ids.input_ids.cuda(),
                            attention_mask=input_ids.attention_mask.cuda(),
                            do_sample=False, num_beams=32, max_new_tokens=256,
                        )[:,input_ids.input_ids.shape[-1]:]
                    output = tokenizer.batch_decode(generated)[0].replace("<|endoftext|>", "").strip()
                    print(output, file=fout)

        with open("evaluation/common_gen-keys-test.txt", "r") as fin:
            with open("evaluation/tmp-generated-test.txt", "w") as fout:
                iterator = tqdm.tqdm(fin.readlines(), dynamic_ncols=True)
                for line in iterator:
                    keys = "<|endoftext|> " + line.strip() + " ="
                    input_ids = tokenizer(
                        keys, return_tensors="pt",
                    )

                    with torch.autocast(device_type='cuda'):
                        generated = model.generate(
                            input_ids=input_ids.input_ids.cuda(),
                            attention_mask=input_ids.attention_mask.cuda(),
                            do_sample=False, num_beams=32, max_new_tokens=256
                        )[:,input_ids.input_ids.shape[-1]:]
                    output = tokenizer.batch_decode(generated)[0].replace("<|endoftext|>", "").strip()
                    print(output, file=fout)

    elif args.stage == "sample":
        from lemminflect import getAllInflections
        import nltk

        model = GPT2LMHeadModel.from_pretrained("./gpt2_sft_commongen")
        model.cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        common_gen = datasets.load_dataset("common_gen")['train']
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id


        def get_all_inflections(key):
            inflection_sets = set(getAllInflections(key).values())
            inflections = set()
            inflections.add(key)
            for subset in inflection_sets:
                for item in subset:
                    inflections.add(item)
            return inflections

        def oracle(key_inflections, sequence):
            word_sets = set(nltk.word_tokenize(sequence.lower()))
            for key_inflection in key_inflections:
                if len(set.intersection(key_inflection, word_sets)) == 0:
                    return 0
            return 1

        concepts = [None] * (common_gen[-1]["concept_set_idx"] + 1)

        iterator = tqdm.tqdm(common_gen, dynamic_ncols=True)
        for instance in iterator:
            line = " ".join(instance["concepts"])
            if concepts[instance["concept_set_idx"]] is None:
                keys = "<|endoftext|> " + line.strip() + " ="
                key_inflections = [
                    get_all_inflections(key.strip()) for key in instance["concepts"]
                ]
                input_ids = tokenizer(
                    keys, return_tensors="pt",
                )
                generated = model.generate(
                    input_ids=input_ids.input_ids.cuda(),
                    attention_mask=input_ids.attention_mask.cuda(),
                    do_sample=True, num_return_sequences=4, max_new_tokens=256, top_p=0.98
                )

                output = tokenizer.batch_decode(generated[:, input_ids.input_ids.shape[-1]:])
                output = [
                    line.replace("<|endoftext|>", "").strip() for line in output
                ]
                labels = [
                    oracle(key_inflections, line) for line in output
                ]

                # labels = [
                #     0 * oracle(key_inflections, line) for line in output
                # ]
                concepts[instance["concept_set_idx"]] = [line, output, labels]

            concepts[instance["concept_set_idx"]][1].append(instance["target"])
            concepts[instance["concept_set_idx"]][2].append(1)
        import pickle
        pickle.dump(concepts, open("sampled_dataset.pyc", "wb"))
        dataset = {
            "concepts": [],
            "target": [],
            "label": []
        }
        for i, instance in enumerate(concepts):
            if instance is None:
                print(i)
                embed()
                exit()
            for target, label in zip(instance[1], instance[2]):
                dataset["concepts"].append(instance[0])
                dataset["target"].append(target)
                dataset["label"].append(label)
        dataset = datasets.Dataset.from_dict(dataset)
        dataset.save_to_disk("./nado_sample")
    elif args.stage == "nado":
        nado_data = datasets.load_from_disk("./nado_sample")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        dataset = LabeledCommonGen(tokenizer, nado_data)
        reference_model = GPT2LMHeadModel.from_pretrained("./gpt2_sft_commongen", device_map=torch.device("cuda"))
        policy_model = GPT2DiNADOMergeLMHeadModel.from_pretrained("gpt2-large",
                                                                resid_pdrop=0.00,
                                                                embd_pdrop=0.00,
                                                                attn_pdrop=0.00,)
        policy_model.cuda()
        reference_model.cuda()
        policy_model.load_state_dict(reference_model.state_dict(), strict=False)
        policy_model.lm_head.load_state_dict(reference_model.lm_head.state_dict())
        policy_model.save_pretrained("./gpt2_nado_epoch0_commongen")
        policy_model.norm_prediction_head[-1].weight.data.zero_()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size // args.grad_step, shuffle=True, drop_last=True, pin_memory=True,
            num_workers=32
        )
        # def decode(instance):
        #     return {
        #         "string": tokenizer.decode(instance["input_ids"]),
        #         "label": instance["labels"]
        #     }
        # embed(); exit()
        # policy_model.eval()
        reference_model.eval()
        opt = AdamW(lr=2e-6, params=policy_model.parameters(), betas=(0.9, 0.95), weight_decay=0.00)

        # from IPython import embed
        # embed()
        # exit()
        open("nado_log.txt", "w").close()

        for epoch_idx in range(80):
            moving_avg_loss = None
            moving_avg_regloss = None
            iterator = tqdm.tqdm(dataloader, dynamic_ncols=True)
            iter_idx = 0

            for batch in iterator:
                input_ids = batch['input_ids']
                mask = batch['attention_mask']
                efft_length = mask.sum(dim=-1)
                batch_maxlen = efft_length.amax().item() + 1
                mask = batch['label_mask']
                input_ids = input_ids[:, 0:batch_maxlen].cuda()
                mask = mask[:, 0:batch_maxlen-1].cuda()

                model_output = policy_model(
                    input_ids = input_ids,
                    labels = batch['labels'].cuda(),
                    reference_model=reference_model
                )

                if iter_idx % args.grad_step == 0:
                    opt.zero_grad()

                loss = (model_output.loss * mask).sum(dim=-1).mean()
                reg_loss = (model_output.reg_loss * mask).sum(dim=-1).mean()

                ((loss + reg_loss) / args.grad_step).backward()

                if moving_avg_loss is None:
                    moving_avg_loss = loss.detach().item()
                    moving_avg_regloss = reg_loss.detach().item()
                else:
                    moving_avg_loss = 0.95 * moving_avg_loss + 0.05 * loss.detach().item()
                    moving_avg_regloss = 0.95 * moving_avg_regloss + 0.05 * reg_loss.detach().item()

                iter_idx += 1

                if iter_idx % args.grad_step == 0:
                    opt.step()
                    if (iter_idx // args.grad_step) % 10 == 0:
                        iterator.write("NADO Epoch %d-%d: Class Loss %f Reg Loss %f" % (epoch_idx, (iter_idx // args.grad_step), moving_avg_loss, moving_avg_regloss))
                        print("%d\t%d\t%f\t%f" % (epoch_idx, (iter_idx // args.grad_step), moving_avg_loss, moving_avg_regloss), file=open("nado_log.txt", "a"))

            policy_model.save_pretrained("./gpt2_nado_epoch%d_commongen" % (epoch_idx+1))
            # exit()

    elif args.stage == "nado_eval":
        # model = GPT2DiNADOMergeLMHeadModel.from_pretrained("./gpt2_sft_commongen")
        model = GPT2DiNADOMergeLMHeadModel.from_pretrained("./gpt2_nado_epoch%s_commongen" % args.grad_step)
        model.cuda()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        if model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        with open("evaluation/common_gen-keys-dev.txt", "r") as fin:
            with open("evaluation/nado-generated-dev.txt", "w") as fout:
                iterator = tqdm.tqdm(fin.readlines(), dynamic_ncols=True)
                for line in iterator:
                    keys = "<|endoftext|> " + line.strip() + " ="
                    input_ids = tokenizer(
                        keys, return_tensors="pt",
                    )
                    generated = model.generate(
                        input_ids=input_ids.input_ids.cuda(),
                        attention_mask=input_ids.attention_mask.cuda(),
                        do_sample=False, num_beams=32, max_new_tokens=256,
                    )[:,input_ids.input_ids.shape[-1]:]
                    output = tokenizer.batch_decode(generated)[0].replace("<|endoftext|>", "").strip()
                    iterator.write(output)
                    print(output, file=fout)

        with open("evaluation/common_gen-keys-test.txt", "r") as fin:
            with open("evaluation/nado-generated-test.txt", "w") as fout:
                iterator = tqdm.tqdm(fin.readlines(), dynamic_ncols=True)
                for line in iterator:
                    keys = "<|endoftext|> " + line.strip() + " ="
                    input_ids = tokenizer(
                        keys, return_tensors="pt",
                    )

                    generated = model.generate(
                        input_ids=input_ids.input_ids.cuda(),
                        attention_mask=input_ids.attention_mask.cuda(),
                        do_sample=False, num_beams=32, max_new_tokens=256
                    )[:,input_ids.input_ids.shape[-1]:]
                    output = tokenizer.batch_decode(generated)[0].replace("<|endoftext|>", "").strip()
                    iterator.write(output)
                    print(output, file=fout)

    elif args.stage == "nado_ddp":
        from transformers import Trainer, TrainingArguments, TrainerCallback
        class CustomTrainer(Trainer):
            def training_step(self, model: torch.nn.Module, inputs):
                """
                Perform a training step on a batch of inputs.

                Subclass and override to inject custom behavior.

                Args:
                    model (`nn.Module`):
                        The model to train.
                    inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                        The inputs and targets of the model.

                        The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                        argument `labels`. Check your model's documentation for all accepted arguments.

                Return:
                    `torch.Tensor`: The tensor with training loss on this batch.
                """
                model.eval()
                inputs = self._prepare_inputs(inputs)

                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)

                # print(type(self.compute_loss_context_manager()))

                return loss

            def compute_loss(self, model, inputs, return_outputs=False):
                pass

if __name__ == "__main__":
    main()