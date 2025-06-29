#include "pch.h"
#include "Final.h"
#include "BasicDlg.h"
#include "afxdialogex.h"


// CBasicDlg 对话框

IMPLEMENT_DYNAMIC(CBasicDlg, CDialogEx)

CBasicDlg::CBasicDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_BASIC_DIALOG, pParent)
	, m_nFontSize(12)
	, m_nXInterval(1)
	, m_nAxisWidth(1)
	, m_colorAxis(RGB(0, 0, 0))
{
}

CBasicDlg::~CBasicDlg()
{
}

// 字体枚举回调函数
int CALLBACK EnumFontFamProc(ENUMLOGFONT* lpelf, NEWTEXTMETRIC* lpntm, DWORD FontType, LPARAM lParam)
{
	CComboBox* pCombo = (CComboBox*)lParam;
	CString strFontName = lpelf->elfLogFont.lfFaceName;
	// 只添加未添加过的字体
	if (pCombo->FindStringExact(-1, strFontName) == CB_ERR)
		pCombo->AddString(strFontName);
	return 1;
}

void CBasicDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_COMBO_FONT, m_comboFont);
	DDX_Text(pDX, IDC_EDIT_FONTSIZE, m_nFontSize);
	DDV_MinMaxInt(pDX, m_nFontSize, 8, 24);
	DDX_Text(pDX, IDC_EDIT_XINTERVAL, m_nXInterval);
	DDV_MinMaxInt(pDX, m_nXInterval, 1, 10);
	DDX_Text(pDX, IDC_EDIT_AXISWIDTH, m_nAxisWidth);
	DDV_MinMaxInt(pDX, m_nAxisWidth, 1, 5);
}

BEGIN_MESSAGE_MAP(CBasicDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BUTTON_AXISCOLOR, &CBasicDlg::OnBnClickedButtonAxiscolor)
	ON_WM_CTLCOLOR()
END_MESSAGE_MAP()

// CBasicDlg 消息处理程序

BOOL CBasicDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// 初始化字体组合框
	CClientDC dc(this);
	::EnumFontFamilies(dc.GetSafeHdc(), NULL, 
		(FONTENUMPROC)EnumFontFamProc, 
		(LPARAM)&m_comboFont);
	// 选择字体
	int nIndex = m_comboFont.FindStringExact(0, m_strFontName);
	if (nIndex != CB_ERR)
		m_comboFont.SetCurSel(nIndex);
	else if (m_comboFont.GetCount() > 0)
		m_comboFont.SetCurSel(0);
	// 更新控件
	UpdateData(FALSE);
	return TRUE;
}

void CBasicDlg::OnBnClickedButtonAxiscolor()
{
	CColorDialog dlg(m_colorAxis, CC_FULLOPEN | CC_ANYCOLOR);
	if (dlg.DoModal() == IDOK)
	{
		m_colorAxis = dlg.GetColor();
		// 更新说明文本的颜色
		CWnd* pStaticLabel = GetDlgItem(IDC_STATIC);
		if (pStaticLabel)
		{
			pStaticLabel->Invalidate();
		}
		Invalidate();
	}
}

HBRUSH CBasicDlg::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor)
{
	HBRUSH hbr = CDialogEx::OnCtlColor(pDC, pWnd, nCtlColor);
	if (nCtlColor == CTLCOLOR_STATIC)
	{
		CString text;
		pWnd->GetWindowText(text);
		if (text == _T("坐标轴颜色："))
		{
			pDC->SetTextColor(m_colorAxis);
		}
	}
	
	return hbr;
}

void CBasicDlg::OnOK()
{
	// 获取控件数据
	if (!UpdateData(TRUE))
		return;
	// 获取选中的字体
	int nSel = m_comboFont.GetCurSel();
	if (nSel != CB_ERR)
	{
		m_comboFont.GetLBText(nSel, m_strFontName);
	}
	else if (m_comboFont.GetCount() > 0)
	{
		m_comboFont.SetCurSel(0);
		m_comboFont.GetLBText(0, m_strFontName);
	}
	else
	{
		m_strFontName = _T("Arial"); // 默认字体
	}
	CDialogEx::OnOK();
} 