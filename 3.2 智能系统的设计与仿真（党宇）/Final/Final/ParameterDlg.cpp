#include "pch.h"
#include "Final.h"
#include "ParameterDlg.h"
#include "afxdialogex.h"

// CParameterDlg 对话框

IMPLEMENT_DYNAMIC(CParameterDlg, CDialogEx)

CParameterDlg::CParameterDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_PARAMETER_DIALOG, pParent)
	, m_dJ(0.5)
	, m_dB(0.5)
	, m_dA(0.2)
	, m_dTauG(10.0)
	, m_dInitialAngle(0.0)
{
}

CParameterDlg::~CParameterDlg()
{
}

void CParameterDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_J, m_dJ);
	DDV_MinMaxDouble(pDX, m_dJ, 0.01, 1.0);
	DDX_Text(pDX, IDC_EDIT_B, m_dB);
	DDV_MinMaxDouble(pDX, m_dB, 0.01, 2.0);
	DDX_Text(pDX, IDC_EDIT_A, m_dA);
	DDV_MinMaxDouble(pDX, m_dA, 0.0, 1.0);
	DDX_Text(pDX, IDC_EDIT_TAUG, m_dTauG);
	DDV_MinMaxDouble(pDX, m_dTauG, 8.0, 12.0);
	DDX_Text(pDX, IDC_EDIT_INITIALANGLE, m_dInitialAngle);
	DDV_MinMaxDouble(pDX, m_dInitialAngle, 0.0, 90.0);
}

BEGIN_MESSAGE_MAP(CParameterDlg, CDialogEx)
	ON_BN_CLICKED(IDOK, &CParameterDlg::OnBnClickedOk)
END_MESSAGE_MAP()

// CParameterDlg 消息处理程序

BOOL CParameterDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	return TRUE;
}

void CParameterDlg::OnBnClickedOk()
{
	// 验证参数是否在有效范围内
	if (!UpdateData(TRUE))
		return;
	// 额外范围检查（虽然DDV已经检查过，但这里可以提供更友好的错误信息）
	CString strError;
	if (m_dJ < 0.01 || m_dJ > 1.0)
		strError = _T("惯性矩J范围应在0.01~1.0 Kg·m^2之间");
	else if (m_dB < 0.01 || m_dB > 2.0)
		strError = _T("阻尼系数B范围应在0.01~2.0 Nm·s·rad^-1之间");
	else if (m_dA < 0.0 || m_dA > 1.0)
		strError = _T("固定摩擦系数A范围应在0~1 Nm之间");
	else if (m_dTauG < 8.0 || m_dTauG > 12.0)
		strError = _T("重力扭矩τg范围应在8~12 Nm之间");
	else if (m_dInitialAngle < 0.0 || m_dInitialAngle > 90.0)
		strError = _T("初始角度范围应在0~90度之间");
	if (!strError.IsEmpty())
	{
		MessageBox(strError, _T("参数错误"), MB_ICONWARNING | MB_OK);
		return;
	}
	CDialogEx::OnOK();
} 