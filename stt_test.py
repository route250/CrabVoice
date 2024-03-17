
from CrabAI.voice.stt import AudioToText, SttData

def item_dump(no,item:SttData):
    text = ""
    type = SttData.type_to_str(item.typ)
    isvoice = " "
    st_sec = item.start/item.sample_rate
    ed_sec = item.end/item.sample_rate
    content = item.content
    text = f"{no:3d} {type:8s} {st_sec:8.3f} - {ed_sec:8.3f} {content}"
    return text

def main():

    wav_filename='testData/nakagawke01.wav'
    # wav_filename='testData/voice_mosimosi.wav'
    # wav_filename='testData/voice_command.wav'
    model="whisper"

    model = None
    save_dir='logs/wave'

    stt_dict_list:list[SttData] = []
    def seg_callback( item:SttData ):
        stt_dict_list.append(item)
        print( item_dump(len(stt_dict_list), item ))
    print(f"loading...")
    STT:AudioToText = AudioToText( model=model,callback=seg_callback, wave_dir=save_dir )
    #STT.load( filename=wav_filename )
    STT.load( mic='default' )
    print(f"start...")
    STT.start()

    selected = input(" >> ")

if __name__ == "__main__":
    main()