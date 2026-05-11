# AAMI_logbase Demo Mock Business Data

本目录包含 3 张用于 Demo 的模拟业务 CSV。

## 文件说明

1. `mock_production_lots.csv`
   - 模拟 SAP 工单 / production lot
   - 粒度：一个 lot / 一张工单
   - 与设备标准表通过 `line + product_name + actual_start_time + actual_end_time` 关联

2. `mock_product_process_specs.csv`
   - 模拟产品工艺参数和 checklist 点检规则
   - 粒度：一个产品 + 一个点检参数
   - `parameter_name` 对应设备标准表字段名
   - `apply_when` 和 `active_filter_*` 用于控制规则生效条件，避免停机时电流=0被误判为工艺异常

3. `mock_qc_avi_results.csv`
   - 模拟 QC / AVI / 良率结果
   - 粒度：一个 lot 的质量结果
   - 与工单表通过 `lot_id` / `production_number` 关联

## 推荐关联方式

不要把 3 张 CSV 物理合并进设备标准表。建议在 Notebook 中运行时关联：

```text
production_lot
  -> 根据 line + actual_start_time + actual_end_time 截取设备标准表时间窗口
  -> 根据 product_name 读取产品工艺参数规则
  -> 执行自动点检
  -> 根据 lot_id 关联 QC / AVI 结果
  -> 生成 Production Lot Profile / 自动点检报告 / AI 洞察
```

## 重要说明

这些业务数据均为 Demo 模拟数据，不代表客户真实 SAP/QC/SOP 数据。
真实上线时，应替换为客户 SAP 工单、QC/AVI 系统、SOP/FMEA/Control Plan 中的正式数据。
