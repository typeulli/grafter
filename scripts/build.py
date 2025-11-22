from pathlib import Path
import shutil
import re
import os
path_here = Path(__file__).parent

build_target = input("Enter the build target (default: cmake-build-release-visual-studio): ")

path_assets = path_here.parent / "assets"

path_build_grafter = path_here.parent / (build_target if build_target else "cmake-build-release-visual-studio")
if not path_build_grafter.exists():
    print(f"Build target '{path_build_grafter}' does not exist.")
    exit(1)


path_build = path_here / "build"

if path_build.exists():
    for item in path_build.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
path_build.mkdir(parents=True, exist_ok=True)

files_grafter: list[Path] = []
for pattern in ("*.exe", "*.exp", "*.lib", "*.dll", "*.html", "*.js", "*.ico"):
    for file in path_build_grafter.glob(pattern):
        try:
            shutil.copy2(file, path_build / file.name)
            files_grafter.append(file)
            print(file.relative_to(path_build_grafter), end=", ")
        except Exception as e:
            print(f"Could not copy {file}: {e}")
print()

try:
    from os import system
    path_index_html = path_build / "index.html"
    system(f"html-minifier-terser --collapse-whitespace --remove-comments --minify-js true --minify-css true -o {path_index_html} {path_index_html}")
except Exception as e:
    print(f"Could not minify index.html: {e}")
    input("Press Enter to continue...")

nsi = """
!define MUI_ICON "{path_ico}"      ; Icon for the program (16x16 ~ 256x256 pixels)
!define MUI_UNICON "{path_ico}"    ; Icon for the installer (high-resolution, used in Windows Vista and later)

RequestExecutionLevel admin

; ===== Include MUI2 =====
!include "MUI2.nsh"
!include "nsDialogs.nsh"

; ===== Basic Information =====
Outfile "GrafterInstaller.exe"
Name "Grafter"
InstallDir "$PROGRAMFILES\\Grafter"
InstallDirRegKey HKCU "Software\\Grafter" "Install_Dir"
Var __VAR_INITIALIZED
Var DESKTOPSHORTCUT
Var STARTMENUSHORTCUT
Var ADDTOPATH
Var HWND_INSTALL_GRAFTER
Var HWND_DESKTOPSHORTCUT
Var HWND_STARTMENUSHORTCUT
Var HWND_ADDTOPATH






Section "Grafter Core"
    SetOutPath "$INSTDIR"
    ; Copy core files here
{files_grafter}
SectionEnd

Section "Uninstall"
    Delete "$DESKTOP\\Grafter.lnk"
    Delete "$SMPROGRAMS\\Grafter\\Grafter.lnk"
    Delete "$SMPROGRAMS\\Grafter\\Uninstall.lnk"
    RMDir "$SMPROGRAMS\\Grafter"
    RMDir /r "$INSTDIR"
    EnVar::DeleteValue "PATH" "$INSTDIR"
    DeleteRegKey HKCU "Software\\Grafter"
    DeleteRegKey HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Grafter"
SectionEnd




; ===== Welcome Page Setting =====
!define MUI_WELCOMEPAGE_TITLE "Welcome to Grafter Installer"
!define MUI_WELCOMEPAGE_TEXT "This installer will guide you through the installation process."
!insertmacro MUI_PAGE_WELCOME

Function onAddToPathClick
    ${{NSD_GetState}} $HWND_ADDTOPATH $ADDTOPATH
FunctionEnd
Function onAddDesktopShortClick
    ${{NSD_GetState}} $HWND_DESKTOPSHORTCUT $DESKTOPSHORTCUT
FunctionEnd
Function onAddStartMenuShortClick
    ${{NSD_GetState}} $HWND_STARTMENUSHORTCUT $STARTMENUSHORTCUT
FunctionEnd

Function configPage
    !insertmacro MUI_HEADER_TEXT "Grafter Setup Options" "Please select your installation preferences"
    nsDialogs::Create 1018
    Pop $0
    ${{If}} $0 == error
        Abort
    ${{EndIf}}

    ${{NSD_CreateLabel}} 0 0 100% 12u "Grafter Installation Options"
    Pop $0

    ${{NSD_CreateCheckbox}} 0u 12u 100% 12u "Install Grafter"
    Pop $HWND_INSTALL_GRAFTER
    ${{NSD_SetState}} $HWND_INSTALL_GRAFTER ${{BST_CHECKED}}
    System::Call 'User32::EnableWindow(i $HWND_INSTALL_GRAFTER, i 0)'
    


    ${{NSD_CreateCheckbox}} 12u 24u 100% 12u "Add Desktop shortcut"
    Pop $HWND_DESKTOPSHORTCUT
    ${{NSD_OnClick}} $HWND_DESKTOPSHORTCUT onAddDesktopShortClick
    ${{If}} $__VAR_INITIALIZED != 1
        StrCpy $DESKTOPSHORTCUT ${{BST_CHECKED}}
    ${{EndIf}}
    ${{NSD_SetState}} $HWND_DESKTOPSHORTCUT $DESKTOPSHORTCUT


    ${{NSD_CreateCheckbox}} 12u 36u 100% 12u "Add Start Menu shortcut"
    Pop $HWND_STARTMENUSHORTCUT
    ${{NSD_OnClick}} $HWND_STARTMENUSHORTCUT onAddStartMenuShortClick
    ${{If}} $__VAR_INITIALIZED != 1
        StrCpy $STARTMENUSHORTCUT ${{BST_CHECKED}}
    ${{EndIf}}
    ${{NSD_SetState}} $HWND_STARTMENUSHORTCUT $STARTMENUSHORTCUT

    
    ${{NSD_CreateCheckbox}} 0u 48u 100% 12u "Add grafter to system PATH"
    Pop $HWND_ADDTOPATH
    ${{NSD_OnClick}} $HWND_ADDTOPATH onAddToPathClick
    ${{If}} $__VAR_INITIALIZED != 1
        StrCpy $ADDTOPATH ${{BST_UNCHECKED}}
    ${{EndIf}}
    ${{NSD_SetState}} $HWND_ADDTOPATH $ADDTOPATH


    ; Set __VAR_INITIALIZED to 1 to indicate that the configuration page has been initialized
    StrCpy $__VAR_INITIALIZED 1
    nsDialogs::Show
FunctionEnd


Page custom configPage


; ===== UI Configuration (MUI2) =====
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"









; --- Settings before installation ---

Function .onInit
FunctionEnd

Section -AdditionalShortcuts
    ; 바탕화면 바로가기 생성 (GUI 설치 시에만 생성)
    ${{If}} $DESKTOPSHORTCUT == ${{BST_CHECKED}}
        CreateShortCut "$DESKTOP\\Grafter.lnk" "$INSTDIR\\grafter.exe"
    ${{EndIf}}
    ${{If}} $STARTMENUSHORTCUT == ${{BST_CHECKED}}
        CreateDirectory "$SMPROGRAMS\\Grafter"
        CreateShortCut "$SMPROGRAMS\\Grafter\\Grafter.lnk" "$INSTDIR\\grafter.exe"
        CreateShortCut "$SMPROGRAMS\\Grafter\\Uninstall.lnk" "$INSTDIR\\Uninstall.exe"
    ${{EndIf}}
SectionEnd

Section -WriteUninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd

Section -WriteReg
    WriteRegStr HKCU "Software\\Grafter" "Install_Dir" "$INSTDIR"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Grafter" "DisplayName" "Grafter"
    WriteRegStr HKCU "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Grafter" "UninstallString" "$INSTDIR\\Uninstall.exe"
SectionEnd

; Add PATH environment variable
Function .onInstSuccess
    ${{If}} $ADDTOPATH == ${{BST_CHECKED}}
        EnVar::AddValue "PATH" "$INSTDIR"
    ${{EndIf}}
FunctionEnd

"""
nsi = nsi.format(files_grafter="\n".join([
    f'    File "{file.name}"' 
    for file in files_grafter
]),
    path_ico=str(path_assets / "grafter.ico"
))
nsi = re.sub(r'[\u3131-\uD79D]+', '', nsi)

path_nsi = path_build / "grafter.nsi"
path_nsi.write_text(nsi, encoding="utf-8")


os.system(f"\"C:\\Program Files (x86)\\NSIS\\makensis.exe\" {path_nsi}")


path_nsi.unlink()

for file in path_build.iterdir():
    if file.name != "GrafterInstaller.exe":
        try:
            if file.is_file() or file.is_symlink():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        except Exception as e:
            print(f"Could not remove {file}: {e}")