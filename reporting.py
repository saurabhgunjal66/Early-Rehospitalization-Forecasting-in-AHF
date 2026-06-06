import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import streamlit as st


DARK_TEMPLATE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,27,42,0.6)',
    font=dict(color='#90a4ae'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.1)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.1)'),
)


class ReportGenerator:
    """Generates comprehensive reports for AHF prediction system."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.styles = getSampleStyleSheet()

        self.title_style = ParagraphStyle(
            'CustomTitle', parent=self.styles['Heading1'],
            fontSize=18, spaceAfter=30,
            textColor=colors.HexColor('#1f77b4')
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading', parent=self.styles['Heading2'],
            fontSize=14, spaceBefore=20, spaceAfter=10,
            textColor=colors.HexColor('#2c3e50')
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────
    def generate_report(self, report_type, date_from, date_to, format_type='pdf'):
        """Generate report based on type and format."""
        try:
            data = self.get_report_data(date_from, date_to)
            if data is None or len(data) == 0:
                return None

            result = None
            if report_type == 'daily_summary':
                result = self.generate_daily_summary(data, format_type)
            elif report_type == 'weekly_summary':
                result = self.generate_weekly_summary(data, format_type)
            elif report_type == 'monthly_summary':
                result = self.generate_monthly_summary(data, format_type)
            elif report_type == 'high_risk_patients':
                result = self.generate_high_risk_report(data, format_type)
            elif report_type == 'model_performance':
                result = self.generate_model_performance_report(format_type)

            # Track report in database
            if result and hasattr(self.db_manager, 'save_report_record'):
                username = st.session_state.get('username', 'system')
                self.db_manager.save_report_record(
                    report_type=report_type,
                    report_format=format_type.upper(),
                    filename=result.get('filename', 'unknown'),
                    date_from=date_from,
                    date_to=date_to,
                    record_count=len(data),
                    generated_by=username
                )

            return result
        except Exception as e:
            print(f"Error generating report: {e}")
            return None

    def get_report_data(self, date_from, date_to):
        """Fetch data for report generation."""
        try:
            assessments = self.db_manager.get_all_assessments()
            if not assessments:
                return None
            df = pd.DataFrame(assessments)
            df['assessment_date'] = pd.to_datetime(df['assessment_date'])
            if date_from:
                df = df[df['assessment_date'].dt.date >= date_from]
            if date_to:
                df = df[df['assessment_date'].dt.date <= date_to]
            return df
        except Exception as e:
            print(f"Error fetching report data: {e}")
            return None

    def get_recent_reports(self, limit=20):
        """Get list of recently generated reports from database."""
        try:
            if hasattr(self.db_manager, 'get_recent_reports'):
                return self.db_manager.get_recent_reports(limit=limit)
            return []
        except Exception:
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # DAILY SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    def generate_daily_summary(self, data, format_type):
        try:
            summary_stats = self._calc_summary(data)
            if format_type == 'pdf':
                return self.create_pdf_daily_summary(summary_stats, data)
            elif format_type == 'csv':
                return self.create_csv_summary(data)
            elif format_type == 'excel':
                return self.create_excel_summary(data)
        except Exception as e:
            print(f"Error generating daily summary: {e}")
            return None

    def _calc_summary(self, data):
        return {
            'total_assessments': len(data),
            'unique_patients': data['patient_id'].nunique(),
            'high_risk_count': len(data[data['risk_level'] == 'High Risk']),
            'moderate_risk_count': len(data[data['risk_level'] == 'Moderate Risk']),
            'low_risk_count': len(data[data['risk_level'] == 'Low Risk']),
            'avg_risk_score': data['ensemble_probability'].mean(),
            'avg_nt_probnp': data['nt_probnp'].mean() if 'nt_probnp' in data else 0,
            'avg_age': data['age'].mean() if 'age' in data else 0,
            'male_percentage': (data['gender'] == 'Male').mean() * 100 if 'gender' in data else 0,
        }

    def create_pdf_daily_summary(self, summary_stats, data):
        try:
            filename = f"daily_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []

            story.append(Paragraph("AHF Risk Assessment Daily Summary", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
            story.append(Spacer(1, 12))

            summary_data = [
                ['Metric', 'Value'],
                ['Total Assessments', str(summary_stats['total_assessments'])],
                ['Unique Patients', str(summary_stats['unique_patients'])],
                ['High Risk Patients', str(summary_stats['high_risk_count'])],
                ['Moderate Risk Patients', str(summary_stats['moderate_risk_count'])],
                ['Low Risk Patients', str(summary_stats['low_risk_count'])],
                ['Average Risk Score', f"{summary_stats['avg_risk_score']:.1%}"],
                ['Average NT-proBNP', f"{summary_stats['avg_nt_probnp']:.0f} pg/mL"],
                ['Average Age', f"{summary_stats['avg_age']:.1f} years"],
                ['Male Percentage', f"{summary_stats['male_percentage']:.1f}%"],
            ]
            story.append(Paragraph("Summary Statistics", self.heading_style))
            story.append(self._styled_table(summary_data, col_widths=[3*inch, 2*inch],
                                            header_color='#3498db'))
            story.append(Spacer(1, 20))

            high_risk = data[data['risk_level'] == 'High Risk']
            if len(high_risk) > 0:
                story.append(Paragraph("High Risk Patients", self.heading_style))
                hr_data = [['Patient ID', 'Age', 'Risk Score', 'NT-proBNP', 'Time']]
                for _, p in high_risk.head(10).iterrows():
                    hr_data.append([
                        str(p['patient_id']), str(p.get('age', '?')),
                        f"{p['ensemble_probability']:.1%}",
                        f"{p.get('nt_probnp', 0):.0f}",
                        p['assessment_date'].strftime('%H:%M')
                    ])
                story.append(self._styled_table(hr_data,
                    col_widths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 0.9*inch],
                    header_color='#e74c3c', body_color='#fadbd8'))

            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by CardioGuard AI — AHF Prediction System", self.styles['Normal']))
            doc.build(story)
            return {'filename': filename, 'mime_type': 'application/pdf'}
        except Exception as e:
            print(f"Error creating PDF daily summary: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # WEEKLY SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    def generate_weekly_summary(self, data, format_type):
        try:
            data = data.copy()
            data['date'] = data['assessment_date'].dt.date
            daily_stats = data.groupby('date').agg(
                total_assessments=('patient_id', 'count'),
                avg_risk=('ensemble_probability', 'mean'),
                risk_std=('ensemble_probability', 'std'),
                high_risk_count=('risk_level', lambda x: (x == 'High Risk').sum())
            ).reset_index()
            if format_type == 'pdf':
                return self.create_weekly_pdf(daily_stats, data)
            else:
                return self.create_csv_summary(data)
        except Exception as e:
            print(f"Error generating weekly summary: {e}")
            return None

    def create_weekly_pdf(self, daily_stats, data):
        try:
            filename = f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []

            story.append(Paragraph("Weekly AHF Risk Assessment Summary", self.title_style))
            story.append(Spacer(1, 12))

            d_min = daily_stats['date'].min()
            d_max = daily_stats['date'].max()
            overview = (f"Period: {d_min} to {d_max} · "
                        f"Total Assessments: {len(data)} · "
                        f"High Risk: {len(data[data['risk_level']=='High Risk'])} · "
                        f"Avg Daily: {daily_stats['total_assessments'].mean():.1f}")
            story.append(Paragraph(overview, self.styles['Normal']))
            story.append(Spacer(1, 16))

            daily_data = [['Date', 'Assessments', 'Avg Risk', 'High Risk']]
            for _, row in daily_stats.iterrows():
                daily_data.append([str(row['date']), str(row['total_assessments']),
                                   f"{row['avg_risk']:.1%}", str(row['high_risk_count'])])
            story.append(Paragraph("Daily Breakdown", self.heading_style))
            story.append(self._styled_table(daily_data,
                col_widths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch]))

            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by CardioGuard AI", self.styles['Normal']))
            doc.build(story)
            return {'filename': filename, 'mime_type': 'application/pdf'}
        except Exception as e:
            print(f"Error creating weekly PDF: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # MONTHLY SUMMARY  (previously unimplemented)
    # ──────────────────────────────────────────────────────────────────────────
    def generate_monthly_summary(self, data, format_type):
        try:
            data = data.copy()
            data['week'] = data['assessment_date'].dt.to_period('W').astype(str)
            weekly_stats = data.groupby('week').agg(
                total_assessments=('patient_id', 'count'),
                avg_risk=('ensemble_probability', 'mean'),
                high_risk_count=('risk_level', lambda x: (x == 'High Risk').sum()),
                unique_patients=('patient_id', 'nunique')
            ).reset_index()

            if format_type == 'pdf':
                return self._create_monthly_pdf(weekly_stats, data)
            elif format_type == 'csv':
                return self.create_csv_summary(data)
            else:
                return self.create_excel_summary(data)
        except Exception as e:
            print(f"Error generating monthly summary: {e}")
            return None

    def _create_monthly_pdf(self, weekly_stats, data):
        try:
            filename = f"monthly_summary_{datetime.now().strftime('%Y%m')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            summary = self._calc_summary(data)

            story.append(Paragraph("Monthly AHF Risk Assessment Summary", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Month: {datetime.now().strftime('%B %Y')} · "
                f"Total: {summary['total_assessments']} assessments · "
                f"Unique Patients: {summary['unique_patients']}",
                self.styles['Normal']))
            story.append(Spacer(1, 16))

            month_table_data = [['Week', 'Assessments', 'Unique Patients', 'Avg Risk', 'High Risk']]
            for _, row in weekly_stats.iterrows():
                month_table_data.append([
                    str(row['week']), str(row['total_assessments']),
                    str(row['unique_patients']), f"{row['avg_risk']:.1%}",
                    str(row['high_risk_count'])
                ])
            story.append(Paragraph("Weekly Breakdown", self.heading_style))
            story.append(self._styled_table(month_table_data,
                col_widths=[1.8*inch, 1.1*inch, 1.2*inch, 1*inch, 1*inch]))

            # Summary stats
            stats_data = [
                ['Metric', 'Value'],
                ['Total Assessments', str(summary['total_assessments'])],
                ['Unique Patients', str(summary['unique_patients'])],
                ['High Risk', str(summary['high_risk_count'])],
                ['Moderate Risk', str(summary['moderate_risk_count'])],
                ['Low Risk', str(summary['low_risk_count'])],
                ['Avg Risk Score', f"{summary['avg_risk_score']:.1%}"],
            ]
            story.append(Spacer(1, 16))
            story.append(Paragraph("Monthly Totals", self.heading_style))
            story.append(self._styled_table(stats_data, col_widths=[3*inch, 2*inch]))
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by CardioGuard AI", self.styles['Normal']))
            doc.build(story)
            return {'filename': filename, 'mime_type': 'application/pdf'}
        except Exception as e:
            print(f"Error creating monthly PDF: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # HIGH RISK REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def generate_high_risk_report(self, data, format_type):
        try:
            high_risk_data = data[data['risk_level'] == 'High Risk'].copy()
            if len(high_risk_data) == 0:
                return None
            high_risk_data = high_risk_data.sort_values('ensemble_probability', ascending=False)
            if format_type == 'pdf':
                return self.create_high_risk_pdf(high_risk_data)
            elif format_type == 'csv':
                return self.create_csv_summary(high_risk_data)
            else:
                return self.create_excel_summary(high_risk_data)
        except Exception as e:
            print(f"Error generating high risk report: {e}")
            return None

    def create_high_risk_pdf(self, high_risk_data):
        try:
            filename = f"high_risk_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            story.append(Paragraph("High Risk Patients Report", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Total high-risk patients: {len(high_risk_data)} · "
                f"Avg risk: {high_risk_data['ensemble_probability'].mean():.1%} · "
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self.styles['Normal']))
            story.append(Spacer(1, 16))

            patient_data = [['Patient ID', 'Age', 'Gender', 'Risk Score', 'NT-proBNP', 'Weight', 'EF%']]
            for _, p in high_risk_data.head(20).iterrows():
                patient_data.append([
                    str(p['patient_id']), str(p.get('age', '?')),
                    str(p.get('gender', '?')),
                    f"{p['ensemble_probability']:.1%}",
                    f"{p.get('nt_probnp', 0):.0f}",
                    f"{p.get('weight', 0):.1f}",
                    f"{p.get('ejection_fraction', 0):.0f}"
                ])
            story.append(Paragraph("Patient Details", self.heading_style))
            story.append(self._styled_table(patient_data,
                col_widths=[1.2*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.6*inch],
                header_color='#e74c3c', body_color='#fadbd8'))
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by CardioGuard AI", self.styles['Normal']))
            doc.build(story)
            return {'filename': filename, 'mime_type': 'application/pdf'}
        except Exception as e:
            print(f"Error creating high risk PDF: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL PERFORMANCE REPORT  (previously unimplemented)
    # ──────────────────────────────────────────────────────────────────────────
    def generate_model_performance_report(self, format_type='pdf'):
        try:
            lr_perf = self.db_manager.get_latest_model_performance('logistic_regression') or {}
            xgb_perf = self.db_manager.get_latest_model_performance('xgboost') or {}

            if format_type == 'pdf':
                return self._create_model_performance_pdf(lr_perf, xgb_perf)
            elif format_type == 'csv':
                rows = []
                for model_name, perf in [('Logistic Regression', lr_perf), ('XGBoost', xgb_perf)]:
                    if perf:
                        rows.append({'Model': model_name, **{k: v for k, v in perf.items()
                                                              if k not in ('feature_importance',)}})
                if not rows:
                    return None
                buf = io.StringIO()
                pd.DataFrame(rows).to_csv(buf, index=False)
                fn = f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                return {'filename': fn, 'data': buf.getvalue(), 'mime_type': 'text/csv'}
            else:
                return self.create_excel_summary(pd.DataFrame())
        except Exception as e:
            print(f"Error generating model performance report: {e}")
            return None

    def _create_model_performance_pdf(self, lr_perf, xgb_perf):
        try:
            filename = f"model_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            story.append(Paragraph("Model Performance Report", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                self.styles['Normal']))
            story.append(Spacer(1, 16))

            metrics_map = [
                ('Accuracy', 'accuracy', '{:.3f}'),
                ('AUC-ROC', 'auc', '{:.3f}'),
                ('Sensitivity', 'sensitivity', '{:.3f}'),
                ('Specificity', 'specificity', '{:.3f}'),
                ('Precision', 'precision_score', '{:.3f}'),
                ('Recall', 'recall', '{:.3f}'),
                ('F1-Score', 'f1_score', '{:.3f}'),
                ('PPV', 'ppv', '{:.3f}'),
                ('NPV', 'npv', '{:.3f}'),
            ]

            table_data = [['Metric', 'Logistic Regression', 'XGBoost']]
            for label, key, fmt in metrics_map:
                lr_val = lr_perf.get(key)
                xgb_val = xgb_perf.get(key)
                table_data.append([
                    label,
                    fmt.format(lr_val) if lr_val is not None else 'N/A',
                    fmt.format(xgb_val) if xgb_val is not None else 'N/A',
                ])

            story.append(Paragraph("Performance Metrics Comparison", self.heading_style))
            story.append(self._styled_table(table_data,
                col_widths=[2*inch, 2*inch, 2*inch],
                header_color='#2ecc71'))
            story.append(Spacer(1, 30))
            story.append(Paragraph("Generated by CardioGuard AI", self.styles['Normal']))
            doc.build(story)
            return {'filename': filename, 'mime_type': 'application/pdf'}
        except Exception as e:
            print(f"Error creating model performance PDF: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # CSV / EXCEL HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def create_csv_summary(self, data):
        try:
            filename = f"assessment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_columns = [c for c in [
                'assessment_date', 'patient_id', 'age', 'gender', 'weight',
                'nt_probnp', 'creatinine', 'b_line_score', 'ejection_fraction',
                'ensemble_probability', 'risk_level'
            ] if c in data.columns]
            export_data = data[export_columns].copy()
            if 'assessment_date' in export_data.columns:
                export_data['assessment_date'] = export_data['assessment_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'ensemble_probability' in export_data.columns:
                export_data['ensemble_probability'] = export_data['ensemble_probability'].apply(lambda x: f"{x:.3f}")
            buf = io.StringIO()
            export_data.to_csv(buf, index=False)
            return {'filename': filename, 'data': buf.getvalue(), 'mime_type': 'text/csv'}
        except Exception as e:
            print(f"Error creating CSV summary: {e}")
            return None

    def create_excel_summary(self, data):
        try:
            filename = f"ahf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            buf = io.BytesIO()
            export_columns = [c for c in [
                'assessment_date', 'patient_id', 'age', 'gender', 'weight',
                'nt_probnp', 'creatinine', 'b_line_score', 'ejection_fraction',
                'ensemble_probability', 'risk_level'
            ] if c in data.columns]

            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                if len(data) > 0:
                    summary = self._calc_summary(data)
                    pd.DataFrame({
                        'Metric': list(summary.keys()),
                        'Value': [str(v) for v in summary.values()]
                    }).to_excel(writer, sheet_name='Summary', index=False)

                    data[export_columns].to_excel(writer, sheet_name='Detailed Data', index=False)

                    hr = data[data['risk_level'] == 'High Risk'][export_columns]
                    if len(hr) > 0:
                        hr.to_excel(writer, sheet_name='High Risk Patients', index=False)

            return {
                'filename': filename,
                'data': buf.getvalue(),
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
        except Exception as e:
            print(f"Error creating Excel summary: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    def _styled_table(self, data, col_widths=None, header_color='#3498db',
                      body_color='#f2f3f4'):
        tbl = Table(data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(body_color)),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor(body_color), colors.HexColor('#ffffff')]),
        ]))
        return tbl
