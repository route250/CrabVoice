from enum import Enum

class VoiceState(Enum):
    ST_STOPPED:int = 0
    ST_TALK_ENTRY:int = 10
    ST_TALK_CONVERT_START:int = 11
    ST_TALK_CONVERT_END:int = 12
    ST_TALK_PLAY_START:int = 13
    ST_TALK_PLAY_END:int = 14
    ST_TALK_EXIT:int = 15
    ST_LISTEN:int = 20
    ST_LISTEN_END: int = 21
    ST_STARTED_STT:int = 100