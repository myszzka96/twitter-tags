from google.cloud import texttospeech
import os


class Speaker:
    """Wrapper class around google cloud text-to-speech service.

    Attributes:
        client (google.cloud.texttospeech.TextToSpeechClient): 
            Client that manages requests to google text-to-speech service.

        voice (google.cloud.texttospeech_v1.types.VoiceSelectionParams): 
            Type of voice to play speech audio with.

        audio_config (google.cloud.texttospeech.types.AudioConfig):
            Type of audio file to be returned (e.g. MP3)
    """
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

        # Build the voice request, select the language code ("en-US") 
        # and the ssml with voice gender ("neutral").
        self.voice = texttospeech.types.VoiceSelectionParams(
            language_code='en-US', 
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
        )

        self.audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.MP3)


    def text_to_speech_wrapper(self, text_input):
        """
        Wrapper function calls google text-to-speech service and gets audio. 

        Args:
            text_input (str): Text to be synthesized into audio file.

        Returns:
            response (google.cloud.texttospeech_v1.types.SynthesizeSpeechResponse): 
                Audio file in binary format. Not sure what is returned if service fails.
        """
        synthesis_input = texttospeech.types.SynthesisInput(text=text_input)
        response = self.client.synthesize_speech(synthesis_input, self.voice, self.audio_config)
        
        return response


    def play_audio(self, audio_content):
        """
        Write audio content into a file, play the file, and then delete it.

        Args:
            audio_content (str): Audio data to be played.

        Returns:
            bool: True if we succeeded in playing audio. False otherwise.
        """

        # File is temporarily in directory where Speaker.py resides.
        file_name = 'output.mp3'  
        try:
            with open(file_name, 'wb') as out:
                out.write(audio_content)
            
            # Play output mp3 file and then delete.
            print('Audio content written to file "%s"' % file_name)
            os.system('mpg123 ' + file_name)
            os.remove(file_name)
            return True
        except IOError as e:
            print 'speak() failed: %s' % e       
        
        return False


    def speak(self, text_input):
        """
        Play given text input as synthesized speech audio.

        Args:
            text_input (str): Text to be played as speech audio.

        Returns:
            bool: True if we succeeded in playing audio. False otherwise.
        """
        response = self.text_to_speech_wrapper(text_input)
        is_success = self.play_audio(response.audio_content)

        return is_success
