// MainFrm.cpp: CMainFrame 类的实现
//

#include "pch.h"
#include "framework.h"
#include "Final.h"
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,       
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

// CMainFrame 构造/析构

CMainFrame::CMainFrame() noexcept
{
}

CMainFrame::~CMainFrame()
{
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
    if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
        return -1;
    if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP | CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) || !m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
    {
        TRACE0("未能创建工具栏\n");
        return -1;    
    }
    if (!m_wndStatusBar.Create(this))
    {
        TRACE0("未能创建状态栏\n");
        return -1;    
    }
    m_wndStatusBar.SetIndicators(indicators, sizeof(indicators)/sizeof(UINT));
    // 设置工具栏按钮文本
    m_wndToolBar.SetButtonText(m_wndToolBar.CommandToIndex(ID_BUTTON_START), _T("开始"));
    m_wndToolBar.SetButtonText(m_wndToolBar.CommandToIndex(ID_BUTTON_PAUSE), _T("暂停/继续"));
    m_wndToolBar.SetButtonText(m_wndToolBar.CommandToIndex(ID_BUTTON_STOP), _T("结束"));
    // 设置工具栏按钮风格
    m_wndToolBar.GetToolBarCtrl().SetExtendedStyle(TBSTYLE_EX_DRAWDDARROWS | TBSTYLE_EX_MIXEDBUTTONS);
    DWORD dwStyle = m_wndToolBar.GetBarStyle();
    dwStyle |= CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC;
    m_wndToolBar.SetBarStyle(dwStyle);
    // 设置快捷键
    HACCEL hAccel = ::LoadAccelerators(AfxGetInstanceHandle(), MAKEINTRESOURCE(IDR_MAINFRAME));
    // 调整工具栏大小
    CRect rectToolBar;
    m_wndToolBar.GetItemRect(0, &rectToolBar);
    int buttonHeight = rectToolBar.Height();
    m_wndToolBar.SetSizes(CSize(rectToolBar.Width(), buttonHeight), CSize(16, 16));
    // 设置工具栏停靠
    m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
    EnableDocking(CBRS_ALIGN_ANY);
    DockControlBar(&m_wndToolBar);
    // 重新停靠工具栏
    CRect rect;
    m_wndToolBar.GetWindowRect(&rect);
    rect.OffsetRect(0, 1);
    DockControlBar(&m_wndToolBar, AFX_IDW_DOCKBAR_TOP, &rect);
    return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	return TRUE;
}

BOOL CMainFrame::OnCreateClient(LPCREATESTRUCT lpcs, CCreateContext* pContext)
{
    // 直接返回默认实现
    return CFrameWnd::OnCreateClient(lpcs, pContext);
}

// CMainFrame 诊断

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}
#endif //_DEBUG


// CMainFrame 消息处理程序

void CMainFrame::OnSize(UINT nType, int cx, int cy)
{
	CFrameWnd::OnSize(nType, cx, cy);
	// 当窗口大小改变时，刷新工具栏
	if (::IsWindow(m_wndToolBar.GetSafeHwnd()))
	{
		// 重新停靠工具栏，让它计算正确大小
		CRect rect;
		m_wndToolBar.GetWindowRect(&rect);
		// 记录工具栏的当前位置
		RepositionBars(AFX_IDW_CONTROLBAR_FIRST, AFX_IDW_CONTROLBAR_LAST, 0);
	}
}