from lm_eval import evaluator 
from lm_eval.models.huggingface import HFLM

def eval_few_shot(model, tokenizer, task_list, num_fewshot=None, limit=3500, apply_chat_template=False):
    model = HFLM(
        pretrained=model, 
        backend="causal",
        tokenizer=tokenizer, 
        max_length=None, 
        batch_size="auto:3", 
        add_bos_token=True,
    )
    
    results = evaluator.simple_evaluate(
        model=model,
        model_args=None,
        tasks=task_list,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        use_cache=None,
        limit=limit,
        write_out=True,
        log_samples=True,
        cache_requests=True,
        rewrite_requests_cache=False,
        apply_chat_template=apply_chat_template,
    )

    return results 