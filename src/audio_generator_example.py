import torch

from tqdm import tqdm

from audio.generate_audio import AudioGenerator
from soundtrack.generate_soundtrack import SoundTrackGenerator

from utils.utils import merge_images_audio_to_video, combine_audio_with_soundtrack

AUDIO_MODEL_NAME = "facebook/mms-tts-por"
SOUNDTRACK_MODEL_NAME = "facebook/musicgen-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    print("Generating text to speech - audio...")
    
    prompts = {
        "question": "Você sabia que existe um animal no Brasil capaz de cavar buracos gigantes em poucos minutos?",
        "introduction": "Um ser noturno, raro e misterioso, que pode chegar a pesar quase sessenta quilos. Esse é o tatú-canastra, o maior tatú do mundo.",
        "desc_fisica": "O tatú-canastra pode atingir até um metro e meio de comprimento contando a cauda. Suas enormes garras, afiadas como lâminas, são capazes de abrir o solo como se fosse papel.",
        "habito_noturno": "Ele passa a maior parte do tempo escondido. Durante a noite, sai para caçar formigas e cupins, seus principais alimentos. Mas o que mais impressiona é a velocidade com que cava túneis, que podem ultrapassar 5 metros de profundidade.",
        "raridade": "Pouco se sabe sobre o tatú-canastra. É um animal extremamente difícil de ser visto na natureza. Muitos moradores de áreas rurais passam a vida inteira sem se deparar com ele. Por isso, é considerado um verdadeiro fantasma da fauna brasileira.",
        "ecologia": "Os buracos deixados pelo tatú-canastra servem de abrigo para dezenas de outros animais. Ou seja, além de enigmático, ele é essencial para o equilíbrio do ecossistema.",
        "ameaca": "Infelizmente, o avanço do desmatamento e a caça reduziram drasticamente sua população. Hoje, o tatú-canastra está ameaçado de extinção, sendo protegido por lei.",
        "encerramento": "O tatú-canastra é uma das criaturas mais misteriosas do Brasil. e talvez você nunca o veja pessoalmente. Mas, se quisermos que ele continue existindo, precisamos proteger o seu habitat. Gostou de conhecer esse animal enigmático?",
        "encer": "Então compartilhe este vídeo, se inscreva no canal e deixe o seu laique. Assim, você nos ajuda a trazer mais histórias sombrias e curiosas da nossa fauna."
    }

    # generate audios for the texts/prompts
    audios = []
    count_token = 0
    for key, prompt in tqdm(prompts.items(), desc="Running audio creation"):
        for token in prompt.split("."):
            if token != "":
                audio_generator = AudioGenerator(
                    model_name=AUDIO_MODEL_NAME,
                    device=DEVICE,
                    output_path=f"{count_token}.wav"
                )
                print(f"Generating for key={key}:{count_token}")
                audio_data = audio_generator.generate(text=token)
                audios.append(audio_data)
                count_token += 1
    
    # combining audios
    output_name = audio_generator.combine(media_list=audios)

    # generate sound track
    print("Generating sound track...")
    sound_generator = SoundTrackGenerator(
        model_name=SOUNDTRACK_MODEL_NAME,
        device=DEVICE,
        output_path="misterious.wav",
        number_tokens=512
    )
    misterious_audio_output = sound_generator.generate(text="A mysterious and curious soundtrack, atmospheric and immersive, with soft rhythmic pulses, subtle textures, and evolving harmonies. No vocals, no sudden changes, just a continuous background mood that feels intriguing and enigmatic.")
    # combining noise audios (1 hour) 
    print("Combining results for misterious audio") 
    soundtrack_output = sound_generator.combine([misterious_audio_output.cpu().numpy().squeeze()] * 720) 

    # combining audio with soundtrack
    output_track = combine_audio_with_soundtrack(
        main_audio_file=output_name,
        soundtrack_file=soundtrack_output,
        output_file="final_audio.wav"
    )

    # compite final audio with images
    images = [
        "tatu-1.jpeg",
        "tatu-2.jpeg",
        "tatu-3.jpeg",
        "tatu-4.jpeg",
        "tatu-5.jpeg"
    ]

    print("Merging final audio to video images")
    merge_images_audio_to_video(
        image_files=images, 
        audio_file="final_audio.wav", 
        output_file="tatu-canastra-final.mp4"
    )