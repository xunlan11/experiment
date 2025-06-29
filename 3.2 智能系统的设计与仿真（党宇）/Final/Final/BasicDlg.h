#pragma once

// CBasicDlg 对话框
class CBasicDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CBasicDlg)

public:
	CBasicDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CBasicDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_BASICG_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	CComboBox m_comboFont;
	int m_nFontSize;
	int m_nXInterval;
	int m_nAxisWidth;
	COLORREF m_colorAxis;
	CString m_strFontName;
	
	virtual BOOL OnInitDialog();
	virtual void OnOK();
	afx_msg void OnBnClickedButtonAxiscolor();
	afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
}; 