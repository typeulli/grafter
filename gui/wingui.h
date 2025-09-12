
#ifndef GRAFTER_WINGUI_H
#define GRAFTER_WINGUI_H
#include <windows.h>
#include <wil/com.h>
#include <WebView2.h>
#include <string>
#include <wrl.h>

using namespace Microsoft::WRL;

int main() {
    // COM 초기화
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    // WebView 생성
    ComPtr<IWebView2Environment> webviewEnvironment;
    CreateWebView2EnvironmentWithDetails(
        nullptr, nullptr, nullptr,
        Callback<IWebView2CreateWebView2EnvironmentCompletedHandler>(
            [](HRESULT result, IWebView2Environment* env) -> HRESULT {
                env->CreateWebView(
                    GetConsoleWindow(), // 부모 윈도우 핸들
                    Callback<IWebView2CreateWebViewCompletedHandler>(
                        [](HRESULT result, IWebView2WebView* webview) -> HRESULT {
                            webview->Navigate(L"https://www.google.com");
                            return S_OK;
                        }).Get());
                return S_OK;
            }).Get());

    // 메시지 루프
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    CoUninitialize();
    return 0;
}
#endif //GRAFTER_WINGUI_H