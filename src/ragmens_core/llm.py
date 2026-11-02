"""Adapter Ollama con streaming, senza side effect all'import."""
import httpx

class LLMError(RuntimeError): pass

class OllamaClient:
    def __init__(self,config,client=None):
        self.config=config
        if client is None:
            import ollama
            client=ollama.Client(host=config.ollama_base_url,timeout=config.llm_timeout)
        self.client=client
    def _request(self,messages,stream):
        c=self.config
        try:
            return self.client.chat(model=c.ollama_model,messages=messages,stream=stream,options={'temperature':c.llm_temperature,'num_ctx':c.llm_num_ctx,'num_predict':c.llm_num_predict,'repeat_penalty':c.llm_repeat_penalty})
        except (httpx.HTTPError,ConnectionError,TimeoutError,ValueError) as exc: raise LLMError(f'Errore Ollama: {exc}') from exc
        except Exception as exc:
            # Il client Ollama usa proprie eccezioni ResponseError e RequestError.
            import ollama
            if isinstance(exc,(ollama.ResponseError,ollama.RequestError)): raise LLMError(f'Errore Ollama: {exc}') from exc
            raise
    @staticmethod
    def _content(response):
        try: return response['message']['content']
        except (TypeError,KeyError,IndexError) as exc: raise LLMError('Risposta Ollama malformata') from exc
    def __call__(self,messages):
        content=self._content(self._request(messages,False))
        if not isinstance(content,str) or not content.strip(): raise LLMError('Risposta Ollama vuota')
        return content.strip()
    def stream(self,messages):
        stream=self._request(messages,True)
        completed=False
        try:
            for chunk in stream:
                text=self._content(chunk)
                if not isinstance(text,str): raise LLMError('Risposta streaming non testuale')
                if chunk.get('done',False): completed=True
                if text: yield text
            if not completed: raise LLMError('Stream Ollama terminato senza conferma di completamento')
        except (httpx.HTTPError,ConnectionError,TimeoutError) as exc: raise LLMError(f'Stream Ollama interrotto: {exc}') from exc
        except Exception as exc:
            import ollama
            if isinstance(exc,(ollama.ResponseError,ollama.RequestError)): raise LLMError(f'Stream Ollama interrotto: {exc}') from exc
            raise
        finally:
            close=getattr(stream,'close',None)
            if close: close()
    def close(self):
        close=getattr(self.client,'close',None)
        if close: close()
        else:
            transport=getattr(self.client,'_client',None)
            if transport is not None: transport.close()

def call_llm(messages,config): return OllamaClient(config)(messages)
