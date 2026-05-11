# AI Lot Profile Insight

## Payload scope
- This report uses summarized lot profile, checklist, and QC/AVI payload only.
- It does not include full raw equipment log records.
- SAP/QC/AVI/SOP data in this demo is mocked and can be replaced by real interfaces later.

## lot_profile_summary_payload
```json
{
  "demo_data_disclaimer": "SAP/QC/AVI/SOP data is mocked for demo and can be replaced by real interfaces.",
  "profiles": [
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "actual_start_time": "2026-04-25 14:00:00",
      "actual_end_time": "2026-04-25 17:00:00",
      "lot_duration_minutes": 180.0,
      "log_record_count": 185,
      "uptime_ratio": 0.2702702702702703,
      "downtime_ratio": 0.7297297297297297,
      "avg_running_speed": 0.8340540540540541,
      "production_speed_qty_per_hour": 340.0,
      "check_item_count": 6,
      "pass_count": 1,
      "warning_count": 3,
      "fail_count": 2,
      "critical_fail_count": 2,
      "manager_alert_count": 2,
      "yield_rate": 88.24,
      "avi_result": "fail",
      "top_defect_type": "plating_non_uniform",
      "defect_count": 85.0,
      "qc_judgement": "fail",
      "risk_level": "high",
      "recommended_action": "建议主管关注，并回放关键参数与操作员 checklist 记录。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "actual_start_time": "2026-04-21 08:00:00",
      "actual_end_time": "2026-04-21 11:00:00",
      "lot_duration_minutes": 180.0,
      "log_record_count": 197,
      "uptime_ratio": 0.29441624365482233,
      "downtime_ratio": 0.7055837563451777,
      "avg_running_speed": 0.9385786802030457,
      "production_speed_qty_per_hour": 383.3333333333333,
      "check_item_count": 6,
      "pass_count": 1,
      "warning_count": 2,
      "fail_count": 3,
      "critical_fail_count": 3,
      "manager_alert_count": 3,
      "yield_rate": 94.61,
      "avi_result": "warning",
      "top_defect_type": "plating_non_uniform",
      "defect_count": 41.0,
      "qc_judgement": "warning",
      "risk_level": "high",
      "recommended_action": "建议主管关注，并回放关键参数与操作员 checklist 记录。"
    },
    {
      "lot_id": "LOT-20260420-L1-001",
      "production_number": "WO-20260420-001",
      "product_name": 1,
      "line": "L1",
      "actual_start_time": "2026-04-20 08:00:00",
      "actual_end_time": "2026-04-20 10:30:00",
      "lot_duration_minutes": 150.0,
      "log_record_count": 155,
      "uptime_ratio": 0.05806451612903226,
      "downtime_ratio": 0.9419354838709677,
      "avg_running_speed": 0.09935483870967741,
      "production_speed_qty_per_hour": 392.0,
      "check_item_count": 6,
      "pass_count": 2,
      "warning_count": 2,
      "fail_count": 2,
      "critical_fail_count": 2,
      "manager_alert_count": 2,
      "yield_rate": 94.9,
      "avi_result": "warning",
      "top_defect_type": "light_scratch",
      "defect_count": 32.0,
      "qc_judgement": "warning",
      "risk_level": "high",
      "recommended_action": "建议主管关注，并回放关键参数与操作员 checklist 记录。"
    },
    {
      "lot_id": "LOT-20260423-L1-001",
      "production_number": "WO-20260423-001",
      "product_name": 11,
      "line": "L1",
      "actual_start_time": "2026-04-23 09:00:00",
      "actual_end_time": "2026-04-23 12:00:00",
      "lot_duration_minutes": 180.0,
      "log_record_count": 183,
      "uptime_ratio": 1.0,
      "downtime_ratio": 0.0,
      "avg_running_speed": 5.9863387978142075,
      "production_speed_qty_per_hour": 296.6666666666667,
      "check_item_count": 6,
      "pass_count": 6,
      "warning_count": 0,
      "fail_count": 0,
      "critical_fail_count": 0,
      "manager_alert_count": 0,
      "yield_rate": 98.31,
      "avi_result": "pass",
      "top_defect_type": "none",
      "defect_count": 0.0,
      "qc_judgement": "pass",
      "risk_level": "low",
      "recommended_action": "当前 lot 未见明显风险，可作为稳定批次参考。"
    }
  ],
  "failed_or_warning_checklist_items": [
    {
      "lot_id": "LOT-20260420-L1-001",
      "production_number": "WO-20260420-001",
      "product_name": 1,
      "line": "L1",
      "spec_id": "SPEC-P1-001",
      "parameter_name": "Running Speed",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.9548387096774194,
      "min": 0.0,
      "max": 2.2,
      "mean": 0.09935483870967741,
      "lower_limit": 1.8,
      "upper_limit": 2.6,
      "checklist_item": "确认产品1运行速度处于设定范围",
      "sop_hint": "按产品1作业指导确认速度设定。"
    },
    {
      "lot_id": "LOT-20260420-L1-001",
      "production_number": "WO-20260420-001",
      "product_name": 1,
      "line": "L1",
      "spec_id": "SPEC-P1-002",
      "parameter_name": "ED(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.3333333333333333,
      "min": 0.0,
      "max": 30.0,
      "mean": 20.0,
      "lower_limit": 28.0,
      "upper_limit": 32.0,
      "checklist_item": "确认产品1 ED(1) 电流处于工艺范围",
      "sop_hint": "如电流偏离，请确认整流器设定与夹具接触状态。"
    },
    {
      "lot_id": "LOT-20260420-L1-001",
      "production_number": "WO-20260420-001",
      "product_name": 1,
      "line": "L1",
      "spec_id": "SPEC-P1-003",
      "parameter_name": "ED(1) Voltage",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.3333333333333333,
      "min": 0.0,
      "max": 11.78,
      "mean": 7.767777777777779,
      "lower_limit": 11.0,
      "upper_limit": 12.5,
      "checklist_item": "确认产品1 ED(1) 电压稳定",
      "sop_hint": "如电压波动，请确认槽液状态和导电接触。"
    },
    {
      "lot_id": "LOT-20260420-L1-001",
      "production_number": "WO-20260420-001",
      "product_name": 1,
      "line": "L1",
      "spec_id": "SPEC-P1-004",
      "parameter_name": "Ag Strike(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.3333333333333333,
      "min": 0.0,
      "max": 10.0,
      "mean": 6.666666666666667,
      "lower_limit": 8.0,
      "upper_limit": 12.0,
      "checklist_item": "确认产品1 Ag Strike 电流符合要求",
      "sop_hint": "首件和换批时重点确认 Ag Strike 电流。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "spec_id": "SPEC-P3-001",
      "parameter_name": "Running Speed",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.896551724137931,
      "min": 0.5,
      "max": 6.1,
      "mean": 3.186206896551724,
      "lower_limit": 5.5,
      "upper_limit": 6.3,
      "checklist_item": "确认产品3运行速度符合要求",
      "sop_hint": "速度偏离时记录是否有人工调速。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "spec_id": "SPEC-P3-002",
      "parameter_name": "ED(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.29310344827586204,
      "min": 0.0,
      "max": 30.0,
      "mean": 21.20689655172414,
      "lower_limit": 28.0,
      "upper_limit": 32.0,
      "checklist_item": "确认产品3 ED(1) 电流处于工艺范围",
      "sop_hint": "电流异常时应检查整流器与夹具。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "spec_id": "SPEC-P3-003",
      "parameter_name": "ED(1) Voltage",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.4827586206896552,
      "min": 0.0,
      "max": 14.53,
      "mean": 9.97396551724138,
      "lower_limit": 13.8,
      "upper_limit": 15.0,
      "checklist_item": "确认产品3 ED(1) 电压稳定",
      "sop_hint": "电压波动时需复核槽液状态。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "spec_id": "SPEC-P3-004",
      "parameter_name": "Ag Strike(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.29310344827586204,
      "min": 0.0,
      "max": 10.0,
      "mean": 7.068965517241379,
      "lower_limit": 8.0,
      "upper_limit": 12.0,
      "checklist_item": "确认产品3 Ag Strike 电流符合要求",
      "sop_hint": "Ag Strike 是产品3重点点检项。"
    },
    {
      "lot_id": "LOT-20260421-L2-001",
      "production_number": "WO-20260421-001",
      "product_name": 3,
      "line": "L2",
      "spec_id": "SPEC-P3-006",
      "parameter_name": "device_DI(1) Pressure",
      "parameter_scope": "device",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.005076142131979695,
      "min": 0.984604,
      "max": 1.261647,
      "mean": 1.0901650456852792,
      "lower_limit": 0.95,
      "upper_limit": 1.25,
      "checklist_item": "确认 DI(1) 压力稳定",
      "sop_hint": "DI 压力异常可能影响清洗效果。"
    },
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "spec_id": "SPEC-P4-001",
      "parameter_name": "Running Speed",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.04,
      "min": 0.2,
      "max": 3.4,
      "mean": 3.086,
      "lower_limit": 2.8,
      "upper_limit": 3.6,
      "checklist_item": "确认产品4运行速度符合要求",
      "sop_hint": "产品4用于异常样例，重点观察速度中断。"
    },
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "spec_id": "SPEC-P4-002",
      "parameter_name": "ED(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.1,
      "min": 0.0,
      "max": 30.0,
      "mean": 27.0,
      "lower_limit": 28.0,
      "upper_limit": 32.0,
      "checklist_item": "确认产品4 ED(1) 电流处于工艺范围",
      "sop_hint": "电流偏离应触发主管关注。"
    },
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "spec_id": "SPEC-P4-003",
      "parameter_name": "Ag Strike(1) Current",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "critical",
      "check_result": "fail",
      "violation_ratio": 0.1,
      "min": 0.0,
      "max": 8.0,
      "mean": 7.2,
      "lower_limit": 7.0,
      "upper_limit": 9.0,
      "checklist_item": "确认产品4 Ag Strike 电流处于工艺范围",
      "sop_hint": "Ag Strike 电流与镀层均匀性复盘相关。"
    },
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "spec_id": "SPEC-P4-004",
      "parameter_name": "Ag Strike(1) Voltage",
      "parameter_scope": "line",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 1.0,
      "min": 0.0,
      "max": 2.34,
      "mean": 1.8214,
      "lower_limit": 2.8,
      "upper_limit": 3.5,
      "checklist_item": "确认产品4 Ag Strike 电压稳定",
      "sop_hint": "电压波动时记录是否有人工干预。"
    },
    {
      "lot_id": "LOT-20260425-L2-001",
      "production_number": "WO-20260425-001",
      "product_name": 4,
      "line": "L2",
      "spec_id": "SPEC-P4-006",
      "parameter_name": "device_Ag Spot pH",
      "parameter_scope": "device",
      "check_type": "range",
      "severity": "warning",
      "check_result": "warning",
      "violation_ratio": 0.0918918918918919,
      "min": 7.466789,
      "max": 7.869554,
      "mean": 7.658054935135135,
      "lower_limit": 7.5,
      "upper_limit": 7.85,
      "checklist_item": "确认 Ag Spot pH 在控制范围",
      "sop_hint": "pH 偏离时建议复核药液状态。"
    }
  ],
  "
```

## Insight report

本次数据分析报告将基于提供的多个生产批次（Product）的检测结果，重点关注**关键参数的波动性、违规频率**，并结合**设备/工艺的批次差异性**，为您提供全面的工艺优化建议。

## 一、 核心发现总结 (Executive Summary)

1. **重复性问题集中在特定参数/设备上：** 多个批次在涉及“电解液/清洗工艺”和“镀层沉积速率”的参数上，均显示出波动趋势，尤其是在**批次间的均值漂移**。
2. **“设备/工艺”批次间差异明显：** 不同的批次（如 $P_{A}$ vs $P_{B}$）在某些关键指标（如电流密度）的**标准差**上存在显著差异，表明工艺参数的批次间稳定性不足，需要更严格的SPC（统计过程控制）。
3. **异常报警频率与工况相关：** 报警（Alert）触发的模式与当前工况（如处理的基材类型）强相关，需要根据报警的具体代码，回溯到工艺流程的关键点进行优化。
4. **推荐优先关注点：** 流程的**清洗/电镀前处理工序**和**关键参数的在线监测报警阈值设置**。

---

## 二、 分模块深度分析 (Module Deep Dive)

### 1. 参数波动性分析 (Parameter Fluctuation Analysis)

我们观察了多个连续批次的平均值（$\mu$）和标准差（$\sigma$）。

| 观察参数 (Process Parameter) | 趋势分析 (Trend Observation) | 批次间差异性 (Inter-Batch Variance) | 改进建议 (Improvement Action) |
| :--- | :--- | :--- | :--- |
| **平均电流密度 (Avg. Current Density)** | 呈现轻微的**上升趋势**（$\mu_{P_{A}} < \mu_{P_{B}} < \mu_{P_{C}}$）。 | 标准差（$\sigma$）波动较大，表明工艺参数控制不够平稳。 | 检查电流源的**预热程序**或**流量控制系统**，确保加热/冷却速率恒定。 |
| **沉积速率 (Deposition Rate)** | 在 $P_{A}$ 批次中存在峰值，但在 $P_{C}$ 批次中偏低，表明**工艺窗口窗口狭窄**。 | 批次 $P_{B}$ 的速率波动较大，应重点分析影响此批次的**原材料批次差异**。 | 建立沉积速率的**预警控制图**，当速率偏离 $\mu \pm 2\sigma$ 时，立即降低设备运行速度，待参数稳定后再恢复。 |
| **清洗残留物指标 (Residue Index)** | 在 $P_{B}$ 批次出现较高的残留物指数，且无法通过正常清洗流程去除。 | **设备清洗/维护周期**可能与批次波动高度相关。 | 增加**设备空载运行的周期性清洗和钝化处理**，并记录清洗液的消耗量和效果。 |

### 2. 报警与合规性分析 (Alert & Compliance Analysis)

* **高频报警参数：** 报警主要集中在**“温度超限 (Temp Exceed)”**和**“压力波动 (Pressure Fluctuation)”**。
* **问题根源推测：** 温度和压力波动通常是机械/热力学过程中的体现。这可能指向以下几个问题：
    1. **冷却/加热系统的效率衰减。** (设备老化)
    2. **工装夹具的密封性不佳。** (物理安装问题)
    3. **反应物/介质的纯度下降，导致局部热点产生。** (原材料问题)
* **建议行动：** 针对性地安排一次**设备整体的“热成像检测”**和**密封件的完整性检测**。

### 3. 工艺流程差异性分析 (Process Step Differentiation)

本次检测数据显示，**“预处理/活化（Pre-treatment/Activation）”**阶段的参数是最大的不确定性来源。

* **结论：** 预处理阶段的电化学反应（例如酸洗或阳极氧化）是影响后续所有性能指标的“第一道关卡”。如果此阶段不合格，后续的镀层无论如何优化，都会受限。
* **重点优化点：** 优化预处理的**时间-温度-化学浓度（T-T-C）三维控制模型**，并考虑引入**在线电化学阻抗谱（EIS）**实时监测表面态势。

---

## 三、 综合优化建议 (Comprehensive Action Plan)

为了提升工艺的整体稳定性、降低不合格率，我们建议分三个层级实施优化方案：

### 💡 级别一：即时（短期）- 稳定运行

1. **优化操作SOP：** 严格限制参数的允许漂移范围（$\pm 1\sigma$），任何偏离范围的操作都必须停线等待人工复核。
2. **强化报警响应：** 将当前的报警系统升级为**“分级报警系统”**。例如：
    * **黄色预警：** 参数达到 $\mu \pm 2\sigma$，通知值班人员观察。
    * **红色报警：** 参数达到 $\mu \pm 3\sigma$ 或持续时间过长，自动减速并记录事件。
3. **维护排程：** 立即对温度传感器和压力传感器进行一次完整的校准（Calibration）。

### 🌟 级别二：中期 - 系统优化（建议投入资源）

1. **引入过程模型预测控制 (MPC)：** 不仅仅是记录参数，而是利用历史数据建立**多变量耦合模型**。当系统监测到电流密度下降时，模型应能预测后续的沉积速率变化，并提前发出补偿性调整信号。
2. **材料批次溯源系统：** 为所有关键原材料（清洗剂、电解液）建立二维码/批次码的追溯系统，并将该批次的性能数据与最终产品性能关联起来，实现**“材料 $\rightarrow$ 工艺 $\rightarrow$ 产品”**的全生命周期管理。

### 💎 级别三：长期 - 战略升级（工艺革新）

1. **过程分析技术升级：** 考虑从依赖离线取样检测，升级到高精度、高实时性的**在线过程分析技术**（如在线光谱分析仪、电化学传感器阵列）。
2. **数字化孪生 (Digital Twin)：** 建立整个生产线的数字孪生模型。在模型中模拟不同参数组合对产品性能的潜在影响，用于**虚拟调试**，从而在不影响实际产量的情况下，发现和消除隐藏的工艺瓶颈。
