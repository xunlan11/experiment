#pragma once

// CParameterDlg 对话框
class CParameterDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CParameterDlg)

public:
	CParameterDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CParameterDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_PARAMETER_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	double m_dJ;        // 惯性矩 (0.01~1 Kg·m^2)
	double m_dB;        // 阻尼系数 (0.01~2 Nm·s·rad^-1)
	double m_dA;        // 固定摩擦系数 (0~1 Nm)
	double m_dTauG;     // 重力扭矩 (8~12 Nm)
	double m_dInitialAngle; // 初始角度 (0~90 deg)
	
	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedOk();
}; 