"""lm-evaluation-harness backend that talks to a running sglang server and times every request.

Registers the model type ``sgl_timed``. It keeps the harness's own CLI, tasks, scoring,
``--apply_chat_template`` and ``--log_samples`` (the same fork MARS used to score its models);
only the model call differs: instead of an in-process HF model it posts every prompt to the
server's ``/generate`` endpoint, ONE request at a time, greedy (temperature 0), and records the
completion-token count and wall-clock latency of each request in a side JSON. Accuracy
(from lm-eval) and decode speed (from the side JSON) therefore come from the SAME generations.
This is the backend behind eval/run_eval.sh; the paper's Table 1 numbers were produced with it.

    python eval/lmeval_sgl.py --model sgl_timed \
        --model_args port=30000,ckpt=/path/to/proposer,max_new_tokens=1024,stats=/path/out/speed_stats.json \
        --tasks gsm8k_cot --num_fewshot 0 --apply_chat_template --log_samples --batch_size 1 \
        --output_path /path/out

model_args
    port            port of the running sglang server (host defaults to 127.0.0.1)
    host            server host (optional)
    ckpt            tokenizer path: the proposer checkpoint (used to render the chat template and
                    to tokenize the prompt; the server receives input_ids, never text)
    max_new_tokens  generation cap applied to EVERY task (1024 in the paper), so that no
                    task-level default silently differs between configurations
    stats           where to write the speed JSON (rewritten every 25 requests and at the end)

speed JSON: n, total_tokens, total_sec, tok_per_s_decode (= total_tokens / total_sec, i.e. pooled
over requests, wall-clock per request measured around the HTTP call), mean_tokens, capped
(requests that hit max_new_tokens), max_new_tokens, and one record per request
{tokens, sec, prompt_tokens, finish}.
"""
import json
import os
import time

import requests as http
from transformers import AutoTokenizer
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("sgl_timed")
class SglTimed(LM):
    def __init__(self, port=None, ckpt=None, max_new_tokens=1024, stats=None, host="127.0.0.1", **kwargs):
        super().__init__()
        assert port and ckpt, "model_args need port=<sglang port> and ckpt=<tokenizer path>"
        self.port = int(port)
        self.host = host
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        self.max_new_tokens = int(max_new_tokens)
        self.stats_path = stats
        self.records = []
        self._rank, self._world_size = 0, 1

    # ---- harness interface ----
    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt, enable_thinking=False)

    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device(self):
        return "cpu"

    def loglikelihood(self, requests):
        raise NotImplementedError("sgl_timed is generate_until only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("sgl_timed is generate_until only")

    # ---- generation ----
    def _dump_stats(self):
        if not self.stats_path:
            return
        tot = sum(r["tokens"] for r in self.records)
        sec = sum(r["sec"] for r in self.records)
        out = {"n": len(self.records), "total_tokens": tot, "total_sec": sec,
               "tok_per_s_decode": tot / max(sec, 1e-9),
               "mean_tokens": tot / max(1, len(self.records)),
               "capped": sum(1 for r in self.records if r["finish"] == "length"),
               "max_new_tokens": self.max_new_tokens, "records": self.records}
        tmp = self.stats_path + ".tmp"
        json.dump(out, open(tmp, "w"))
        os.replace(tmp, self.stats_path)

    def generate_until(self, requests, disable_tqdm=False):
        out = []
        for k, req in enumerate(requests):
            context, gen_kwargs = req.args[0], dict(req.args[1] or {})
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
            payload = {"input_ids": ids,
                       "sampling_params": {"temperature": 0.0, "max_new_tokens": self.max_new_tokens,
                                           "stop": list(until)[:8]}}
            t = time.perf_counter()
            r = http.post(f"http://{self.host}:{self.port}/generate", json=payload, timeout=3600)
            dt = time.perf_counter() - t
            r.raise_for_status()
            d = r.json()
            d = d[0] if isinstance(d, list) else d
            text = d.get("text", "")
            for s in until:                      # the harness convention: cut at the stop string
                if s and s in text:
                    text = text.split(s)[0]
            mi = d.get("meta_info", {})
            fr = mi.get("finish_reason", {})
            self.records.append({"tokens": mi.get("completion_tokens") or 0, "sec": dt,
                                 "prompt_tokens": len(ids),
                                 "finish": fr.get("type", "") if isinstance(fr, dict) else str(fr)})
            out.append(text)
            if (k + 1) % 25 == 0 or k + 1 == len(requests):
                tot = sum(x["tokens"] for x in self.records); sec = sum(x["sec"] for x in self.records)
                print(f"[sgl_timed] {k+1}/{len(requests)}  {tot/max(sec,1e-9):.1f} tok/s pooled", flush=True)
                self._dump_stats()
        self._dump_stats()
        return out


if __name__ == "__main__":
    cli_evaluate()
