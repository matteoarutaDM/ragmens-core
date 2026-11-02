from dataclasses import replace
from ragmens_core import ConversationMemory

def test_memory_scope_and_persistence(config,embedder):
    off=ConversationMemory(config,embedder,'c'); off.add_turn('q','a'); assert not off.path.exists()
    cfg=replace(config,use_memory=True,memory_recent_turns=1)
    m=ConversationMemory(cfg,embedder,'c')
    m.add_turn('pompa','risposta',sources=['pompa.txt'],scope=['pompa.txt'])
    m.add_turn('valvola','risposta',sources=['valvola.md'],scope=['valvola.md'])
    assert not m.recall('pompa')['recent']
    assert m.recall('pompa',['pompa.txt'])['relevant'][0]['question']=='pompa'
    reopened=ConversationMemory(cfg,embedder,'c')
    assert reopened.recall('valvola',['valvola.md'])['recent']
    assert not ConversationMemory(replace(cfg,session_id='altro'),embedder,'c').path.exists()
    assert not ConversationMemory(cfg,embedder,'other').path.exists()
