#pragma once

// PID参数结构体
struct PIDParams
{
	double kp;
	double ki;
	double kd;
	double eval;
};

// CSimulationDlg 对话框
class CSimulationDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CSimulationDlg)

public:
	CSimulationDlg(CWnd* pParent = nullptr);   // 标准构造函数
	virtual ~CSimulationDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SIMULATION_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	double m_dSimTime;      // 仿真时间 (1~100s)
	double m_dTimeStep;     // 仿真步长 (0.001~1s)
	BOOL m_bConstantAngle;  // 目标角度类型：TRUE=定值，FALSE=正弦
	double m_dTargetAngle;  // 目标角度幅值 (0~90deg)
	double m_dSinePeriod;   // 正弦曲线周期 (0.5~10s)
	int m_nSimMethod;       // 仿真方法：0=欧拉法，1=四阶龙格库塔法
	double m_dKp;           // 比例系数
	double m_dKi;           // 积分系数
	double m_dKd;           // 微分系数
	double m_nEval;         // 评价
	CListCtrl m_listPID;    // PID参数列表
	CArray<PIDParams, PIDParams&> m_arrPIDParams;  // PID参数数组
	int m_nSelectedPID;     // 当前选中的PID参数索引

	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedRadioConstant();
	afx_msg void OnBnClickedRadioSine();
	afx_msg void OnBnClickedButtonInsert();
	afx_msg void OnNMClickListPidrecords(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnNMDblclkListPidrecords(NMHDR *pNMHDR, LRESULT *pResult);
	afx_msg void OnBnClickedOk();
}; 