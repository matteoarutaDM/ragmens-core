"""Orchestrazione del motore documentale e del generatore."""
import json
from ragmens_core.config import Config
from ragmens_core.context import build_context, estimate_tokens
from ragmens_core.index import prepare_rag
from ragmens_core.llm import OllamaClient, LLMError
from ragmens_core.memory import ConversationMemory
from ragmens_core.prompts import SYSTEM_PROMPT, build_messages
from ragmens_core.retrieval import retrieve, source_scope

class RAGPipeline:
    def __init__(self,config=None,*,embedder=None,llm=None,system_prompt=SYSTEM_PROMPT,converter=None,client=None):
        self.config=config or Config.from_env()
        self.state=prepare_rag(self.config,embedder=embedder,converter=converter,client=client)
        try:
            self.llm=llm if llm is not None else OllamaClient(self.config)
            self.system_prompt=system_prompt
            self.memory=ConversationMemory(self.config,self.state.embedder,self.state.corpus_signature)
        except BaseException:
            self.state.close()
            raise
    @classmethod
    def prepare(cls,config=None,**kwargs): return cls(config,**kwargs)
    def preview(self,question,*,top_k=None,sources=None):
        hits=retrieve(self.state,question,top_k,sources)
        cfg=self.config
        max_tokens=min(cfg.context_max_tokens,max(0,cfg.llm_num_ctx-cfg.system_reserve_tokens-cfg.memory_reserve_tokens-max(cfg.response_reserve_tokens,cfg.llm_num_predict)-estimate_tokens(question)))
        context,used,excluded,truncated=build_context(hits,max_tokens,cfg.context_max_chars)
        return {'question':question,'hits':hits,'context':context,'sources':used,'excluded':excluded,'truncated':truncated,'token_budget_estimated':max_tokens}
    def _prepare(self,question,top_k,sources,extra_context):
        scope=source_scope(sources)
        result=self.preview(question,top_k=top_k,sources=scope)
        memory=self.memory.context(question,scope)
        messages=build_messages(question,result['context'],memory,self.system_prompt,extra_context)
        if sum(estimate_tokens(json.dumps(m,ensure_ascii=False)) for m in messages)+max(self.config.response_reserve_tokens,self.config.llm_num_predict) > self.config.llm_num_ctx:
            raise ValueError('Prompt stimato oltre il limite del contesto LLM; ridurre memoria, documenti o contesto aggiuntivo')
        return result,messages,scope
    def _save(self,result,answer,scope):
        self.memory.add_turn(result['question'],answer,sources=[x['source'] for x in result['sources']],scope=scope)
    def ask(self,question,*,top_k=None,sources=None,save_memory=True,extra_context=''):
        result,messages,scope=self._prepare(question,top_k,sources,extra_context)
        answer=self.llm(messages)
        if not isinstance(answer,str) or not answer.strip(): raise LLMError('Risposta LLM vuota')
        if save_memory: self._save(result,answer,scope)
        return {**result,'answer':answer,'messages':messages}
    def ask_stream(self,question,*,top_k=None,sources=None,save_memory=True,extra_context=''):
        result,messages,scope=self._prepare(question,top_k,sources,extra_context)
        def generate():
            pieces=[]
            for piece in self.llm.stream(messages):
                pieces.append(piece); yield piece
            answer=''.join(pieces)
            if not answer.strip(): raise LLMError('Risposta streaming vuota')
            if save_memory: self._save(result,answer,scope)
        return generate()
    def close(self):
        try: self.state.close()
        finally:
            close=getattr(self.llm,'close',None)
            if close: close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
