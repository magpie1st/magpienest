# AudioBookTTS XTTS GUI

Coqui XTTS v2 모델을 활용해 오디오북 스타일의 낭독을 생성하는 Python Qt 애플리케이션입니다. 텍스트를 붙여 넣고 합성을 실행하면 파형과 진행 슬라이더를 통해 상태를 확인하고, 결과 음성을 바로 재생하거나 WAV 파일로 저장할 수 있습니다.

## 준비 사항

- Python 3.10 이상 권장 (PySide6, TTS 호환)
- 의존 패키지 설치
  ```bash
  pip install -r requirements.txt
  ```
  - 주요 의존성: `PySide6`, `pyqtgraph`, `numpy`, `soundfile`, `TTS`
- PyTorch/torchaudio 설치 (CUDA 환경에 맞춰 설치 권장)
- 기본 화자 파일은 `D:\Downloads\coqui_voice_pack_v2\voice_pack_v2\voice\my_reader.wav` 경로를 사용합니다. 경로를 바꿨다면 해당 위치로 옮기거나 앱에서 직접 다시 선택하세요.

## 환경 설정

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> GPU를 사용하고 싶다면 `TTS` 설치 전에 CUDA 버전에 맞는 PyTorch를 먼저 설치하세요.

## 사용 방법

```bash
python app.py
```

1. 앱을 실행하면 상단에 텍스트 입력 창, 그 아래에 파형과 재생 슬라이더, 하단에 로그 패널이 표시됩니다. 기본 화자가 자동으로 로드되면 상태 라벨과 로그에 표시됩니다.
2. 필요하면 **Select Speaker** 버튼(또는 메뉴)으로 다른 화자 음성을 선택하세요.
3. 낭독할 영어 텍스트를 입력 창에 붙여 넣습니다.
4. **Synthesize** 버튼을 누르면 XTTS v2 모델이 영어(en)로 합성을 시작합니다. 첫 실행은 모델 다운로드 때문에 시간이 걸릴 수 있습니다.
5. 합성 중에는 로그 패널에서 “Preparing XTTS model…”, “Synthesizing audio…” 메시지를, 파형 아래 슬라이더에서 재생 위치를 확인할 수 있습니다.
6. 합성이 끝나면 파형이 업데이트되고 자동으로 재생이 시작됩니다. 슬라이더를 움직이면 원하는 위치로 이동하며, **Stop** 버튼으로 재생을 멈출 수 있습니다.
7. 텍스트를 수정하지 않은 상태에서 다시 **Synthesize**를 누르면 캐시된 음성을 바로 재생하므로 불필요한 재합성이 발생하지 않습니다.
8. **Save Audio** 버튼으로 결과를 WAV 파일로 저장합니다.

앱을 종료하면 재생을 위해 생성된 임시 WAV 파일은 자동으로 삭제됩니다.

## 추가 팁

- 로그 패널과 재생 슬라이더를 통해 다운로드/합성 흐름과 재생 위치를 GUI 안에서 바로 확인하고 조정할 수 있습니다.
- PowerShell 또는 CMD에서 `python app.py`를 실행하면 콘솔 로그도 함께 확인할 수 있습니다.
- 기본 화자와 언어는 `DEFAULT_SPEAKER_PATH`, `DEFAULT_LANGUAGE` 상수를 수정해 바꿀 수 있습니다.
- 기본 화자 파일을 찾지 못하면 합성이 시작되지 않으니 경로를 다시 확인하거나 직접 화자를 선택하세요.
- 저장된 WAV 파일은 오디오 편집기나 플레이어에서 바로 재생할 수 있습니다.
