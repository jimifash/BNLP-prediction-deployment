import pandas as pd

def get_template():
   
    columns = [
        'Loyalty_Program_Status',
        'Gender',
        'CustomerSatisfaction',
        'Installment_Number',
        'Total_Installments',
        'Installment_Amount',
        'Principal_Amount',
        'Interest_Amount',
        'Amount_Due',
        'Currency',
        'Payment_Method',
        'Order Status',
        'Delay_Days',
        'Repayment_Status',
        'Total_Loans',
        'Segment',
        'Quantity',
        'Unit Price',
        'Total Amount',
        'Discount Applied (%)',
        'Tax (%)',
        'Payment_Mode',
        'Month Order Count',
        'Avg Qantity',
        'Cancelled Order Per Customer',
        'Order Channel',
        'ProductCategory',
        'ProductBrand',
        'ProductCategory2',
        'Cost Price',
        'Discount (%)',
        'PurchaseFrequency',
        'Stock Status',
        'Product Status',
        'Customers_Location',
        'Customer_Duration',
        'Age_Group',
        'installment_to_principal_ratio',
        'interest_to_amount_due_ratio',
        'remaining_installments',
        'amount_per_unit',
        'loans_per_purchase_cycle'
    ]
    
    # Create and return empty dataframe
    return pd.DataFrame(columns=columns)
