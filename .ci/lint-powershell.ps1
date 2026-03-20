$ErrorActionPreference = 'Stop'

$settings = @{
    Severity = @(
        'Information',
        'Warning',
        'Error'
    )
    IncludeDefaultRules = $true
    # Additional rules that are disabled by default
    Rules = @{
        PSAvoidExclaimOperator = @{
            Enable = $true
        }
        PSAvoidLongLines = @{
            Enable = $true
            MaximumLineLength = 120
        }
        PSAvoidSemicolonsAsLineTerminators = @{
            Enable = $true
        }
        PSPlaceCloseBrace = @{
            Enable = $true
            NoEmptyLineBefore = $true
            IgnoreOneLineBlock = $true
            NewLineAfter = $false
        }
        PSPlaceOpenBrace = @{
            Enable = $true
            OnSameLine = $true
            NewLineAfter = $true
            IgnoreOneLineBlock = $true
        }
        PSUseConsistentIndentation = @{
            Enable = $true
            IndentationSize = 4
            PipelineIndentation = 'IncreaseIndentationAfterEveryPipeline'
            Kind = 'space'
        }
        PSUseConsistentWhitespace = @{
            Enable = $true
            CheckInnerBrace = $true
            CheckOpenBrace = $true
            CheckOpenParen = $true
            CheckOperator = $true
            CheckSeparator = $true
            CheckPipe = $true
            CheckPipeForRedundantWhitespace = $true
            CheckParameter = $true
            IgnoreAssignmentOperatorInsideHashTable = $false
        }
        PSUseCorrectCasing = @{
            Enable = $true
        }
    }
}

Invoke-ScriptAnalyzer -Path ./ -Recurse -EnableExit -Settings $settings
