#include "pch.h"
#include "Final.h"
#include "SimulationDlg.h"
#include "afxdialogex.h"

// CSimulationDlg 对话框

IMPLEMENT_DYNAMIC(CSimulationDlg, CDialogEx)

CSimulationDlg::CSimulationDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_SIMULATION_DIALOG, pParent)
	, m_dSimTime(10.0)
	, m_dTimeStep(0.01)
	, m_bConstantAngle(TRUE)
	, m_dTargetAngle(30.0)
	, m_dSinePeriod(2.0)
	, m_nSimMethod(0)
	, m_dKp(10.0)
	, m_dKi(5.0)
	, m_dKd(2.0)
	, m_nEval(0)
	, m_nSelectedPID(-1)
{
}

CSimulationDlg::~CSimulationDlg()
{
}

void CSimulationDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_SIMTIME, m_dSimTime);
	DDV_MinMaxDouble(pDX, m_dSimTime, 1.0, 100.0);
	DDX_Text(pDX, IDC_EDIT_TIMESTEP, m_dTimeStep);
	DDV_MinMaxDouble(pDX, m_dTimeStep, 0.001, 1.0);
	DDX_Radio(pDX, IDC_RADIO_CONSTANT, m_bConstantAngle);
	DDX_Text(pDX, IDC_EDIT_TARGETANGLE, m_dTargetAngle);
	DDV_MinMaxDouble(pDX, m_dTargetAngle, 0.0, 90.0);
	DDX_Text(pDX, IDC_EDIT_SINEPERIOD, m_dSinePeriod);
	DDV_MinMaxDouble(pDX, m_dSinePeriod, 0.5, 10.0);
	DDX_CBIndex(pDX, IDC_COMBO_METHOD, m_nSimMethod);
	DDX_Text(pDX, IDC_EDIT_KP, m_dKp);
	DDX_Text(pDX, IDC_EDIT_KI, m_dKi);
	DDX_Text(pDX, IDC_EDIT_KD, m_dKd);
	DDX_Text(pDX, IDC_STATIC_ERROR, m_nEval);
	DDX_Control(pDX, IDC_LIST_PIDRECORDS, m_listPID);
}

BEGIN_MESSAGE_MAP(CSimulationDlg, CDialogEx)
	ON_BN_CLICKED(IDC_RADIO_CONSTANT, &CSimulationDlg::OnBnClickedRadioConstant)
	ON_BN_CLICKED(IDC_RADIO_SINE, &CSimulationDlg::OnBnClickedRadioSine)
	ON_BN_CLICKED(IDC_BUTTON_INSERT, &CSimulationDlg::OnBnClickedButtonInsert)
	ON_NOTIFY(NM_CLICK, IDC_LIST_PIDRECORDS, &CSimulationDlg::OnNMClickListPidrecords)
	ON_NOTIFY(NM_DBLCLK, IDC_LIST_PIDRECORDS, &CSimulationDlg::OnNMDblclkListPidrecords)
	ON_BN_CLICKED(IDOK, &CSimulationDlg::OnBnClickedOk)
END_MESSAGE_MAP()

// CSimulationDlg 消息处理程序

BOOL CSimulationDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	// 初始化列表控件
	m_listPID.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
	m_listPID.InsertColumn(0, _T("Kp"), LVCFMT_LEFT, 60);
	m_listPID.InsertColumn(1, _T("Ki"), LVCFMT_LEFT, 60);
	m_listPID.InsertColumn(2, _T("Kd"), LVCFMT_LEFT, 60);
	m_listPID.InsertColumn(3, _T("评价"), LVCFMT_LEFT, 80);
	// 初始化仿真方法组合框
	CComboBox* pCombo = (CComboBox*)GetDlgItem(IDC_COMBO_METHOD);
	pCombo->AddString(_T("欧拉法"));
	pCombo->AddString(_T("四阶龙格库塔法"));
	pCombo->SetCurSel(0);
	
	// 根据目标角度类型设置周期控件的可见性
	// 由于m_bConstantAngle在构造函数中被初始化为TRUE（定值模式）
	// 因此这里需要禁用周期设置控件
	GetDlgItem(IDC_EDIT_SINEPERIOD)->EnableWindow(FALSE);
	GetDlgItem(IDC_STATIC_PERIOD)->EnableWindow(FALSE);
	
	// 加载PID参数记录到列表控件
	for (int i = 0; i < m_arrPIDParams.GetSize(); i++)
	{
		CString strItem;
		strItem.Format(_T("%.2f"), m_arrPIDParams[i].kp);
		int nItem = m_listPID.InsertItem(i, strItem);
		strItem.Format(_T("%.2f"), m_arrPIDParams[i].ki);
		m_listPID.SetItemText(nItem, 1, strItem);
		strItem.Format(_T("%.2f"), m_arrPIDParams[i].kd);
		m_listPID.SetItemText(nItem, 2, strItem);
		strItem.Format(_T("%d"), m_arrPIDParams[i].eval);
		m_listPID.SetItemText(nItem, 3, strItem);
	}
	return TRUE;
}

void CSimulationDlg::OnBnClickedRadioConstant()
{
	m_bConstantAngle = TRUE;
	GetDlgItem(IDC_EDIT_SINEPERIOD)->EnableWindow(FALSE);
	GetDlgItem(IDC_STATIC_PERIOD)->EnableWindow(FALSE);
}

void CSimulationDlg::OnBnClickedRadioSine()
{
	m_bConstantAngle = FALSE;
	GetDlgItem(IDC_EDIT_SINEPERIOD)->EnableWindow(TRUE);
	GetDlgItem(IDC_STATIC_PERIOD)->EnableWindow(TRUE);
}

void CSimulationDlg::OnBnClickedButtonInsert()
{
	// 获取当前PID参数
	UpdateData(TRUE);
	// 检查参数
	if (m_dKp < 0 || m_dKi < 0 || m_dKd < 0)
	{
		MessageBox(_T("PID参数不能为负数"), _T("参数错误"), MB_ICONWARNING | MB_OK);
		return;
	}
	// 获取评价
	int eval = 0;
	CString strEval;
	// 获取误差评价输入框的值
	CWnd* pErrorEdit = GetDlgItem(IDC_STATIC_ERROR);
	if (pErrorEdit != NULL)
	{
		pErrorEdit->GetWindowText(strEval);
		if (!strEval.IsEmpty())
		{
			eval = _ttoi(strEval);
		}
	}
	// 添加到参数数组
	PIDParams params;
	params.kp = m_dKp;
	params.ki = m_dKi;
	params.kd = m_dKd;
	params.eval = eval;
	m_arrPIDParams.Add(params);
	// 添加到列表控件
	int nItem = m_listPID.GetItemCount();
	CString strItem;
	strItem.Format(_T("%.2f"), m_dKp);
	m_listPID.InsertItem(nItem, strItem);
	strItem.Format(_T("%.2f"), m_dKi);
	m_listPID.SetItemText(nItem, 1, strItem);
	strItem.Format(_T("%.2f"), m_dKd);
	m_listPID.SetItemText(nItem, 2, strItem);
	strItem.Format(_T("%d"), eval);
	m_listPID.SetItemText(nItem, 3, strItem);
	m_listPID.SetItemState(nItem, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
	m_nSelectedPID = nItem;
}

void CSimulationDlg::OnNMClickListPidrecords(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMITEMACTIVATE pNMItemActivate = reinterpret_cast<LPNMITEMACTIVATE>(pNMHDR);
	// 获取选中项
	m_nSelectedPID = pNMItemActivate->iItem;
	if (m_nSelectedPID >= 0 && m_nSelectedPID < m_arrPIDParams.GetSize())
	{
		// 更新界面显示
		m_dKp = m_arrPIDParams[m_nSelectedPID].kp;
		m_dKi = m_arrPIDParams[m_nSelectedPID].ki;
		m_dKd = m_arrPIDParams[m_nSelectedPID].kd;
		UpdateData(FALSE);
	}
	*pResult = 0;
}

void CSimulationDlg::OnNMDblclkListPidrecords(NMHDR *pNMHDR, LRESULT *pResult)
{
	LPNMITEMACTIVATE pNMItemActivate = reinterpret_cast<LPNMITEMACTIVATE>(pNMHDR);
	// 获取双击项
	int nItem = pNMItemActivate->iItem;
	if (nItem >= 0 && nItem < m_arrPIDParams.GetSize())
	{
		// 从列表和数组中删除
		m_listPID.DeleteItem(nItem);
		m_arrPIDParams.RemoveAt(nItem);
		// 如果删除了选中项，重置选中索引
		if (m_nSelectedPID == nItem)
			m_nSelectedPID = -1;
		else if (m_nSelectedPID > nItem)
			m_nSelectedPID--;
	}
	*pResult = 0;
}

void CSimulationDlg::OnBnClickedOk()
{
	// 验证参数是否在有效范围内
	if (!UpdateData(TRUE))
		return;
	// 额外范围检查
	CString strError;
	if (m_dSimTime < 1.0 || m_dSimTime > 100.0)
		strError = _T("仿真时间范围应在1~100秒之间");
	else if (m_dTimeStep < 0.001 || m_dTimeStep > 1.0)
		strError = _T("仿真步长范围应在0.001~1秒之间");
	else if (m_dTargetAngle < 0.0 || m_dTargetAngle > 90.0)
		strError = _T("目标角度幅值范围应在0~90度之间");
	else if (!m_bConstantAngle && (m_dSinePeriod < 0.5 || m_dSinePeriod > 10.0))
		strError = _T("正弦曲线周期范围应在0.5~10秒之间");
	if (!strError.IsEmpty())
	{
		MessageBox(strError, _T("参数错误"), MB_ICONWARNING | MB_OK);
		return;
	}
	CDialogEx::OnOK();
} 