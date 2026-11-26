# coding: utf-8
"""Script for generating files with NuGet package metadata."""

import datetime
import sys
from pathlib import Path
from shutil import copyfile

if __name__ == "__main__":
    source = Path(sys.argv[1])
    nuget_dir = Path(__file__).absolute().parent / "nuget"
    print(f"Creating nuget directory '{nuget_dir}'")
    libraries = {
        "linux-x64": ("lib_lightgbm.so", "lib_lightgbm.so"),
        "linux-arm64": ("lib_lightgbm_arm64.so", "lib_lightgbm.so"),
        "osx-x64": ("lib_lightgbm.dylib", "lib_lightgbm.dylib"),
        "osx-arm64": ("lib_lightgbm_arm64.dylib", "lib_lightgbm.dylib"),
        "win-x64": ("lib_lightgbm.dll", "lib_lightgbm.dll"),
        "win-arm64": ("lib_lightgbm_arm64.dll", "lib_lightgbm.dll"),
    }
    for filename in [*(source_name for source_name, _ in libraries.values()), "lightgbm.exe"]:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"Required NuGet input is missing: {source / filename}")
    build_folder_path = nuget_dir / "build"
    build_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Looking for libraries in '{source}'")
    for rid, (source_name, target_name) in libraries.items():
        native_dir = nuget_dir / "runtimes" / rid / "native"
        native_dir.mkdir(parents=True, exist_ok=True)
        copyfile(source / source_name, native_dir / target_name)
    copyfile(source / "lightgbm.exe", nuget_dir / "runtimes" / "win-x64" / "native" / "lightgbm.exe")
    version = (nuget_dir.parents[1] / "VERSION.txt").read_text(encoding="utf-8").strip().replace("rc", "-rc")
    print(f"Setting version to '{version}'")
    nuget_str = rf"""<?xml version="1.0"?>
    <package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">
    <metadata>
        <id>LightGBM</id>
        <version>{version}</version>
        <authors>Guolin Ke</authors>
        <owners>Guolin Ke</owners>
        <license type="expression">MIT</license>
        <projectUrl>https://github.com/lightgbm-org/LightGBM</projectUrl>
        <requireLicenseAcceptance>false</requireLicenseAcceptance>
        <description>A fast, distributed, high performance gradient boosting framework</description>
        <copyright>Copyright {datetime.datetime.now().year} @ Microsoft</copyright>
        <tags>machine-learning data-mining distributed native boosting gbdt</tags>
        <dependencies> </dependencies>
    </metadata>
        <files>
        <file src="build\**" target="build"/>
        <file src="runtimes\**" target="runtimes"/>
        </files>
    </package>
    """
    prop_str = r"""
    <Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <ItemGroup Condition="Exists('packages.config') OR
                            Exists('$(MSBuildProjectName).packages.config') OR
                            Exists('packages.$(MSBuildProjectName).config')">
        <Content Include="$(MSBuildThisFileDirectory)/../runtimes/win-x64/native/*.dll"
                Condition="'$(PlatformTarget)' == 'x64'">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>false</Visible>
        </Content>
        <Content Include="$(MSBuildThisFileDirectory)/../runtimes/win-x64/native/*.exe"
                Condition="'$(PlatformTarget)' == 'x64'">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>false</Visible>
        </Content>
        <Content Include="$(MSBuildThisFileDirectory)/../runtimes/win-arm64/native/*.dll"
                Condition="'$(PlatformTarget)' == 'ARM64'">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
        <Visible>false</Visible>
        </Content>
    </ItemGroup>
    </Project>
    """
    target_str = r"""
    <Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
    <PropertyGroup>
        <EnableLightGBMUnsupportedPlatformTargetCheck Condition="'$(EnableLightGBMUnsupportedPlatformTargetCheck)' == ''">true</EnableLightGBMUnsupportedPlatformTargetCheck>
    </PropertyGroup>
    <Target Name="_LightGBMCheckForUnsupportedPlatformTarget"
            Condition="'$(EnableLightGBMUnsupportedPlatformTargetCheck)' == 'true'"
            AfterTargets="_CheckForInvalidConfigurationAndPlatform">
        <Error Condition="'$(PlatformTarget)' != 'x64' AND
                        '$(PlatformTarget)' != 'ARM64' AND
                        ('$(OutputType)' == 'Exe' OR '$(OutputType)'=='WinExe') AND
                        !('$(TargetFrameworkIdentifier)' == '.NETCoreApp' AND '$(PlatformTarget)' == '')"
            Text="LightGBM currently supports 'x64' and 'ARM64' processor architectures. Please ensure your application is targeting 'x64' or 'ARM64'." />
    </Target>
    </Project>
    """
    (nuget_dir / "LightGBM.nuspec").write_text(nuget_str, encoding="utf-8")
    (nuget_dir / "build" / "LightGBM.props").write_text(prop_str, encoding="utf-8")
    (nuget_dir / "build" / "LightGBM.targets").write_text(target_str, encoding="utf-8")
    print("Done creating NuGet package")
