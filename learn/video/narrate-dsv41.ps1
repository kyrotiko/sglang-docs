param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$storyboard = Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
[System.IO.Directory]::CreateDirectory($OutputDirectory) | Out-Null
$narrator = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $narrator.SelectVoice($storyboard.voice)
    $narrator.Rate = $storyboard.voiceRate
    $narrator.Volume = 100
    $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
        24000,
        [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
        [System.Speech.AudioFormat.AudioChannel]::Mono
    )
    for ($sceneIndex = 0; $sceneIndex -lt $storyboard.scenes.Count; $sceneIndex++) {
        $destination = Join-Path $OutputDirectory ('voice-{0:d2}.wav' -f $sceneIndex)
        $narrator.SetOutputToWaveFile($destination, $format)
        $narrator.Speak([string]$storyboard.scenes[$sceneIndex].say)
        $narrator.SetOutputToNull()
        Write-Output ('Narration {0}/{1}' -f ($sceneIndex + 1), $storyboard.scenes.Count)
    }
} finally {
    $narrator.Dispose()
}
