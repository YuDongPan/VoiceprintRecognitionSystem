# Voiceprint Recognition System
A comprehensive and user-friendly voiceprint recognition system that supports SHERPA and ECAPA algorithms. It features end-to-end workflows for training data collection, model training, and multi-mode testing/evaluation, with an intuitive graphical user interface (GUI) for seamless operation.

![img](assets/voiceprint_index.png)

## Project Overview
This system simplifies voiceprint recognition tasks through a fully graphical interface, enabling users to:
- Configure the number of training audio samples
- Select between two state-of-the-art voiceprint recognition algorithms (SHERPA/ECAPA)
- Collect voice data for model training
- Train models with one-click operation
- Perform batch testing and quantitative evaluation of trained models
- Generate detailed evaluation reports with key metrics

## Key Features
- **Training Sample Configuration**: Set custom numbers of training audio samples (minimum 3 samples required for both algorithms).
- **Algorithm Selection**: Support for two mainstream voiceprint recognition algorithms with complementary strengths.
- **Real-time Status Monitoring**: Instant display of training sample count and validation of sample adequacy for the selected algorithm.
- **Simplified Model Management**: One-click operations for training data collection, model training, and model testing.
- **Multi-mode Testing**: Support for batch voiceprint verification via "Multi Test" mode.
- **Comprehensive Algorithm Evaluation**: Automatic generation of detailed reports including matching results, similarity scores, threshold values, and accuracy metrics.

## Supported Algorithms
| Algorithm | GitHub Repository  | Minimum Training Samples | Core Capability |
|-----------|--------------------|--------------------------|-----------------|
| SHERPA    | [https://github.com/k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 3 | Robust voiceprint feature extraction and matching |
| ECAPA     | [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain) | 3 | High-precision speaker verification |

## Environment Setup
This project is built with Python 3.10.14. Follow the steps below to install dependencies:

### Step 1: Create `requirements.txt`
Create a file named `requirements.txt` with the following content:
```txt
numpy==1.26.4
pyaudio==0.2.14
PyQT5==5.15.11
librosa==0.11.0
sherpa-onnx==1.12.20
sounddevice==0.5.3
soundfile==0.12.1
speechbrain==1.0.3
pypinyin==0.55.0
loguru==0.7.2
torch==2.1.1+cu121
torchaudio==2.1.1+cu121
```

### Step 2: Install Dependencies
Execute the following commands in your terminal (ensure Conda is installed):
```bash
# Create a Conda environment
conda create -n voice_system python=3.10.14 -y
# Activate the environment
conda activate voice_system
# Install dependencies
pip install -r requirements.txt
```

> Note: If pyaudio installation fails (common on Windows/macOS):
> - macOS: Install PortAudio first via brew install portaudio, then reinstall pyaudio.
> - Windows: Use pip install pipwin && pipwin install pyaudio for precompiled binaries.
> - Linux: Install PortAudio via sudo apt-get install portaudio19-dev, then reinstall pyaudio.


## Usage Workflow
### Step 1: Configure Training Samples
1. Open the "Set Training Audio Count" window.
2. Enter the number of training samples (≥3) and click "Confirm".
3. Click "Cancel" to abandon the configuration.

### Step 2: Collect Training Data
1. Select the target algorithm (SHERPA/ECAPA) in the main interface.
2. Click the "Collect Training Data" button to record voice samples (follow the on-screen prompts).

### Step 3: Train the Model
1. Ensure the training sample count meets the algorithm’s minimum requirement (≥3).
2. Click the "Train Model" button; the system will automatically train the selected algorithm model.
3. The interface will display "Trained" status for the model once completed.

### Step 4: Test the Model
1. Enter the "Multi Test" mode (supports batch testing).
2. Follow the prompt to read the specified text (≈5 seconds per test) and use "Start Recording Test Voice" / "Stop Recording" to capture test audio.
3. Navigate between test items with "Previous" / "Next" buttons.

### Step 5: Evaluate Algorithms
1. Click "Start Evaluate All Trained Algorithms" after completing test recordings.
2. The system will generate a detailed report including:
   - Target user file matching results (True: Match)
   - Non-target user file matching results (True: Mismatch)
   - Algorithm accuracy (e.g., 100.00% in test cases)
   - Detailed metrics (similarity score, threshold value, result correctness)

## Notes
1. Both SHERPA and ECAPA algorithms require **at least 3 training samples**; the system will prompt if samples are insufficient.
2. Test audio recording duration should follow on-screen prompts (≈5 seconds) for optimal recognition accuracy.
3. Evaluation reports include both "match/mismatch" results and quantitative metrics (similarity/threshold) for debugging and optimization.

## Interface Preview
### Main Interface
- Algorithm selection (SHERPA/ECAPA)
- Real-time data/model status display
- Core operation buttons (Collect Training Data / Train Model / Test Model)

![img](assets/main_interface.png)

### Test Interface
- Multi-mode test configuration
- Recording control (Start/Stop/Previous/Next)
- Detailed test/evaluation result list

![img](assets/test_interface.png)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
```text
MIT License

Copyright (c) 2026 Voiceprint Recognition System Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```