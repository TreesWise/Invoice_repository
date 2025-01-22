import re
import logging
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from database import SingletonSQLDatabase  # Import the Singleton connection instance
from custom_datatypes import ModelInput
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from dotenv import load_dotenv
import os
load_dotenv()

# OpenAI API Key


openai_api_key = os.getenv("openai_api_key")


# Initialize FastAPI application
app = FastAPI()

# Function to keep the database connection alive
def keep_connection_alive():
    try:
        db = SingletonSQLDatabase.get_instance()  # Get the singleton database instance
        db.run("SELECT 1")  # Execute a simple query to keep the connection alive
        logging.info("Database connection kept alive.")
    except Exception as e:
        logging.error("Error in keep_connection_alive:", exc_info=True)

# Initialize APScheduler
scheduler = BackgroundScheduler()

# Schedule the keep_connection_alive task to run every 10 seconds
scheduler.add_job(keep_connection_alive, 'interval', seconds=999999)

# Function to get the database connection via dependency injection
def get_db_connection():
    db = SingletonSQLDatabase.get_instance()
    return db

# The main query handler function
@app.post("/query/")
async def handle_query(userinput: ModelInput, db: SQLDatabase = Depends(get_db_connection)) -> Dict:
    
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            streaming=True,
            verbose=False,
            openai_api_key=openai_api_key
        )

        # Initialize the SQLDatabaseToolkit with LLM and the database
        
        toolkit = SQLDatabaseToolkit(llm=llm, db=db)
        dialect = toolkit.dialect
        top_k = 10

        # Construct the prompt with the provided user input
        column_metadata = """

        **Description:**
        This table contains cleansed and structured item data for equipment and parts used in various systems. It provides detailed information on item descriptions, specifications, manufacturer details, and unique identifiers to ensure accurate item tracking and data integrity.

        **Columns Metadata:**

        - **vessel_id**: Unique identifier for the vessel.
        - **vessel_object_id**: Internal object identifier for the vessel in the system.
        - **Vessel_Name**: Name of the vessel associated with the purchase order or invoice.
        - **Invoice_Id**: Unique identifier for the invoice.
        - **Inv_Title**: Title or description of the invoice.
        - **Inv_Code**: Unique code assigned to the invoice.
        - **VENDOR_INVOICE_DATE**: Date the vendor issued the invoice.
        - **INVOICE_CURRENCY_ID**: Unique identifier for the currency used in the invoice.
        - **Invoice_Currency_Name**: Name of the currency used in the invoice (e.g., US Dollar, Euro).
        - **Invoice_Currency_Code**: ISO currency code (e.g., USD, EUR) for the invoice.
        - **EXG_RATE_VESSEL_CURRENCY**: Exchange rate to convert the invoice currency into the vessel's operating currency.
        - **EXG_RATE_GROUP_CURRENCY**: Exchange rate to convert the invoice currency into the group's base currency.
        - **REGISTRATION_DATE**: Date the invoice was registered in the system.
        - **inv_approval_flag**: Indicates if the invoice has been approved (1 for approved, 0 for pending).
        - **CASH_PO_INVOICE**: Indicates if the invoice is linked to a cash-based purchase order.
        - **VENDOR_INVOICE_NO**: Unique number assigned to the invoice by the vendor.
        - **PARTIAL_INVOICE**: Indicates if the invoice is a partial payment or installment (1 for partial).
        - **PAID_ADV**: Amount of advance payment made against the invoice.
        - **INVOICE_TYPE**: Type of invoice (e.g., Regular, Credit, Debit).
        - **VAT_AMOUNT**: Value-added tax applied to the invoice.
        - **INVOICE_AMOUNT**: Total amount of the invoice.
        - **CASH_DISCOUNT**: Discount applied for early payment in cash.
        - **PAY_DUE_DATE**: Due date for payment of the invoice.
        - **INVOICE_YEAR**: Year the invoice was issued.
        - **PAYMENT_DATE**: Date the invoice was paid.
        - **Inv_Dt_Id**: Unique identifier for the invoice date.
        - **Inv_Effective_Date**: Date when the invoice became effective (e.g., for accounting purposes).
        - **Inv_Closed_Date**: Date the invoice was marked as closed.
        - **total_invoice_currency**: Total invoice amount in the original invoice currency.
        - **po_hd_Id**: Unique identifier for the purchase order header.
        - **po_code**: Code for the purchase order.
        - **created_date**: Date the purchase order was created.
        - **po_dt_id**: Unique identifier for the purchase order details.
        - **po_id**: Unique identifier for the purchase order.
        - **po_quantity**: Quantity of items in the purchase order.
        - **po_unit_price**: Unit price of items in the purchase order.
        - **po_account_id**: Account identifier for the purchase order.
        - **effective_date**: Effective date of the purchase order.
        - **closed_date**: Date the purchase order was marked as closed.
        - **lead_days**: Lead time in days for the purchase order.
        - **requisition_dt_id**: Unique identifier for the requisition details.
        - **enquiry_dt_id**: Unique identifier for the enquiry details.
        - **quote_dt_id**: Unique identifier for the quote details.
        - **approved_quote_dt_id**: Unique identifier for the approved quote details.
        - **approval_list_id**: Unique identifier for the approval list.
        - **sp_goods_receipt_id**: Unique identifier for the goods receipt note.
        - **sp_goods_receipt_hd_code**: Header code for the goods receipt note.
        - **grn_receipt_date**: Date the goods receipt was created.
        - **grn_receipt_status**: Status of the goods receipt (e.g., Pending, Approved).
        - **vendor_delivery_reference**: Vendor’s reference for the delivery associated with the invoice.
        - **grn_remarks**: Remarks or notes about the goods receipt.
        - **send_to_office**: Indicates whether the goods receipt is sent to the office.
        - **goods_rcvd_approved_date**: Date the received goods were approved.
        - **sp_goods_receipt_dt_id**: Unique identifier for the goods receipt date.
        - **item_id**: Unique identifier for the item being purchased or invoiced.
        - **uom_id**: Unit of measure for the item (e.g., KG, LTR, PCS).
        - **received_qty**: Quantity of goods received from the vendor.
        - **accepted_qty**: Quantity of goods accepted after inspection.
        - **converted_stock_qty**: Quantity of goods added to inventory after conversion or repackaging.
        - **normal_qty**: Standard quantity for the item in inventory.
        - **reconditioned_qty**: Quantity of goods reconditioned for use.
        - **port_id**: Unique identifier for the port.
        - **port_name**: Name of the port.
        - **port_code**: Code representing the port (e.g., for internal or system use).
        - **port_unloccode**: UN/LOCODE for the port, used for international trade and transport.
        - **ai_port_unloc_id**: AI-enhanced identifier combining the UN/LOCODE with system-specific details.
        - **QUANTITY**: Quantity of the item involved in the transaction.
        - **UNIT_PRICE**: Unit price of the item involved in the transaction.
        - **ACCOUNT_ID**: Identifier for the account linked to the transaction.
        - **ENTITY_ID**: Unique identifier for the entity associated with the transaction.
        - **TOTAL_VESSEL_CURRENCY**: Total amount in the vessel's operating currency.
        - **TOTAL_PO_CURRENCY**: Total amount in the purchase order's currency.
        - **TOTAL_GROUP_CURRENCY**: Total amount in the group's base currency.
        - **Inv_Account_Code**: Code representing the account linked to the invoice.
        - **Inv_Account_Name**: Name of the account linked to the invoice.
        - **Vendor**: Name of the vendor associated with the invoice.
        - **Vendor_ID**: Unique identifier for the vendor.
        - **Account_Number**: Account number associated with the vendor.
        - **BANK_CODE**: Code of the bank where the vendor holds an account.
        - **SWIFT_CODE**: SWIFT code for the vendor's bank.
        - **IBAN**: IBAN (International Bank Account Number) for the vendor.

        """
        Metadata_Groupings= """
        **Description:**
        This table contains cleansed and structured item data for equipment and parts used in various systems. It provides detailed information on item descriptions, specifications, manufacturer details, and unique identifiers to ensure accurate item tracking and data integrity.

        **Columns Metadata:**

        ### Metadata Groupings

        #### Invoice Details:
        - **Invoice_Id**: Unique identifier for the invoice.
        - **Inv_Title**: Title or description of the invoice.
        - **Inv_Code**: Unique code assigned to the invoice.
        - **VENDOR_INVOICE_DATE**: Date the vendor issued the invoice.
        - **INVOICE_CURRENCY_ID**: Unique identifier for the currency used in the invoice.
        - **Invoice_Currency_Name**: Name of the currency used in the invoice (e.g., US Dollar, Euro).
        - **Invoice_Currency_Code**: ISO currency code (e.g., USD, EUR) for the invoice.
        - **EXG_RATE_VESSEL_CURRENCY**: Exchange rate to convert the invoice currency into the vessel's operating currency.
        - **EXG_RATE_GROUP_CURRENCY**: Exchange rate to convert the invoice currency into the group's base currency.
        - **REGISTRATION_DATE**: Date the invoice was registered in the system.
        - **inv_approval_flag**: Indicates if the invoice has been approved (1 for approved, 0 for pending).
        - **VENDOR_INVOICE_NO**: Unique number assigned to the invoice by the vendor.
        - **PARTIAL_INVOICE**: Indicates if the invoice is a partial payment or installment (1 for partial).
        - **PAID_ADV**: Amount of advance payment made against the invoice.
        - **INVOICE_TYPE**: Type of invoice (e.g., Regular, Credit, Debit).
        - **VAT_AMOUNT**: Value-added tax applied to the invoice.
        - **INVOICE_AMOUNT**: Total amount of the invoice.
        - **CASH_DISCOUNT**: Discount applied for early payment in cash.
        - **PAY_DUE_DATE**: Due date for payment of the invoice.
        - **INVOICE_YEAR**: Year the invoice was issued.
        - **PAYMENT_DATE**: Date the invoice was paid.
        - **Inv_Dt_Id**: Unique identifier for the invoice date.
        - **Inv_Effective_Date**: Date when the invoice became effective (e.g., for accounting purposes).
        - **Inv_Closed_Date**: Date the invoice was marked as closed.
        - **total_invoice_currency**: Total invoice amount in the original invoice currency.

        #### Purchase Order Details:
        - **po_hd_Id**: Unique identifier for the purchase order header.
        - **po_code**: Code for the purchase order.
        - **created_date**: Date the purchase order was created.
        - **po_dt_id**: Unique identifier for the purchase order details.
        - **po_id**: Unique identifier for the purchase order.
        - **po_quantity**: Quantity of items in the purchase order.
        - **po_unit_price**: Unit price of items in the purchase order.
        - **po_account_id**: Account identifier for the purchase order.
        - **effective_date**: Effective date of the purchase order.
        - **closed_date**: Date the purchase order was marked as closed.
        - **lead_days**: Lead time in days for the purchase order.

        #### Requisition and Approval Details:
        - **requisition_dt_id**: Unique identifier for the requisition details.
        - **enquiry_dt_id**: Unique identifier for the enquiry details.
        - **quote_dt_id**: Unique identifier for the quote details.
        - **approved_quote_dt_id**: Unique identifier for the approved quote details.
        - **approval_list_id**: Unique identifier for the approval list.

        #### Goods Receipt Note (GRN) Details:
        - **sp_goods_receipt_id**: Unique identifier for the goods receipt note.
        - **sp_goods_receipt_hd_code**: Header code for the goods receipt note.
        - **grn_receipt_date**: Date the goods receipt was created.
        - **grn_receipt_status**: Status of the goods receipt (e.g., Pending, Approved).
        - **vendor_delivery_reference**: Vendor’s reference for the delivery associated with the invoice.
        - **grn_remarks**: Remarks or notes about the goods receipt.
        - **send_to_office**: Indicates whether the goods receipt is sent to the office.
        - **goods_rcvd_approved_date**: Date the received goods were approved.
        - **sp_goods_receipt_dt_id**: Unique identifier for the goods receipt date.

        #### Item and Quantity Details:
        - **item_id**: Unique identifier for the item being purchased or invoiced.
        - **uom_id**: Unit of measure for the item (e.g., KG, LTR, PCS).
        - **received_qty**: Quantity of goods received from the vendor.
        - **accepted_qty**: Quantity of goods accepted after inspection.
        - **converted_stock_qty**: Quantity of goods added to inventory after conversion or repackaging.
        - **normal_qty**: Standard quantity for the item in inventory.
        - **reconditioned_qty**: Quantity of goods reconditioned for use.

        #### Port Details:
        - **port_id**: Unique identifier for the port.
        - **port_name**: Name of the port.
        - **port_code**: Code representing the port (e.g., for internal or system use).
        - **port_unloccode**: UN/LOCODE for the port, used for international trade and transport.
        - **ai_port_unloc_id**: AI-enhanced identifier combining the UN/LOCODE with system-specific details.

        #### Financial Details:
        - **QUANTITY**: Quantity of the item involved in the transaction.
        - **UNIT_PRICE**: Unit price of the item involved in the transaction.
        - **ACCOUNT_ID**: Identifier for the account linked to the transaction.
        - **ENTITY_ID**: Unique identifier for the entity associated with the transaction.
        - **TOTAL_VESSEL_CURRENCY**: Total amount in the vessel's operating currency.
        - **TOTAL_PO_CURRENCY**: Total amount in the purchase order's currency.
        - **TOTAL_GROUP_CURRENCY**: Total amount in the group's base currency.
        - **Inv_Account_Code**: Code representing the account linked to the invoice.
        - **Inv_Account_Name**: Name of the account linked to the invoice.

        #### Vendor Details:
        - **Vendor**: Name of the vendor associated with the invoice.
        - **Vendor_ID**: Unique identifier for the vendor.
        - **Account_Number**: Account number associated with the vendor.
        - **BANK_CODE**: Code of the bank where the vendor holds an account.
        - **SWIFT_CODE**: SWIFT code for the vendor's bank.
        - **IBAN**: IBAN (International Bank Account Number) for the vendor.

        """
  
       
        prefix = """
        You are an advanced SQL database assistant specializing in answering user queries by interacting with the `Vw_Ai_Tbl_PO_PurchaseOrders_Invoices_Details` table in the `Common` schema.
        ### Handling General Queries:
        - If the query is a general greeting (e.g., "Hi", "Hello", "How are you?"), respond with a polite acknowledgment:
          - Example: "Hello! How can I assist you today?"
        - For unrelated or unclear questions, politely guide the user back to database-specific queries.
          - Example: "I'm here to assist with database-related queries. How can I help?"

        ### Responsibilities:
        1. Provide **precise** and **contextually relevant** answers strictly based on the specified table and schema.
        2. Ensure **query normalization and standardization** to deliver consistent and meaningful results for similar questions.
        3. Leverage response history to avoid redundant queries, optimizing efficiency and user satisfaction.
        
        ### Query Normalization Guidelines:
        - **Input Transformation**: 
        1.Convert all input text to lowercase for case-insensitive handling.
        2.Replace punctuation characters (e.g., -, _, ,, .) with spaces for better uniformity.
        3.Remove leading and trailing whitespaces; collapse multiple spaces into a single space.
        - **String Functions**:
        1.Use SQL string functions like `LOWER()`, `TRIM()`, `REPLACE()`, and fuzzy matching (`LIKE`, `LEVENSHTEIN()`, `SOUNDEX`) to account for minor spelling errors or variations.
        - **Case Mismatch Handling**:
        1.If the data in the database is stored in a specific case (e.g., uppercase), ensure that both the input and the database column are transformed to the same case during processing.
        - **For consistent matching**: 
        1.Normalize input to match the stored case (e.g., UPPER() for uppercase or LOWER() for lowercase).
        2.Apply the same transformation on both sides of the comparison.
        3.Use case-insensitive comparisons (e.g., ILIKE for PostgreSQL, collations in MySQL).
        ### SQL Query Construction:
        1. Ensure the query adheres to the **{dialect} dialect** syntax.
        2. Use **specific columns** in the SELECT clause for precision; avoid `SELECT *`.
        3. Apply **LIMIT {top_k}** unless the user explicitly specifies a different limit in their query. The value of `top_k` is dynamically set to **30** by default for this session, ensuring the response includes at most 30 results unless overridden by user input.
        - If no explicit limit is mentioned in the query, default to **LIMIT {top_k}**, where `top_k=30` for this session.
        - If the user explicitly specifies a `LIMIT` value, override the default `top_k` and use the user's provided value.
        - Ensure every SQL query includes a `LIMIT` clause, either with the default `top_k` or as explicitly stated by the user.
        - Priority for `LIMIT`:
          1. Explicit value provided by the user.
          2. Default value set to `top_k=30` for this session.
        4. Order results by **relevant columns** for clarity (e.g., `ApprovedDate DESC` for recent approvals).
        5. Validate query syntax before execution to ensure success and eliminate errors.
        6. Incorporate conditions for **filtering by user intent** and domain-specific logic (e.g., fetching purchase orders for a particular `VesselName` or `SMC`).
        7. When queried regarding **unique vendors**, the unique vendors are supposed to be calculated based on  **VENDOREMAIL**
        8. Use the **PO_USD_VALUE** column for the questions regarding the purchase.
        ### Rules of Engagement:
        - Do not perform Data Manipulation Language (DML) operations such as `INSERT`, `UPDATE`, or `DELETE`.
        - Use **Markdown format** for presenting results:
          - Include bordered tables for tabular data for better readability.
        - If the query is unrelated to the database or cannot be addressed, respond with:
          *"I'm unable to provide an answer for that. This information is not available."*
        - Handle ambiguous questions by:
          1. Politely clarifying the user's intent.
          2. Assuming the most logical interpretation when clarification isn't feasible.
        - **Tone and Style**:
          - Be professional, concise, and courteous in responses.
          - Avoid database-specific jargon unless directly relevant.
          - Use the following metadata {column_metadata} and {Metadata_Groupings}
        
        Your ultimate goal is to ensure clarity, accuracy, and user satisfaction while adhering strictly to data access and usage guidelines.


        """
        
       

        
        suffix = """
        If asked about the database structure, table design, or unavailable data, respond politely:
        *"I can answer questions from this database but cannot provide information about its structure or column names. Let me assist you with the data instead."*
        
        ### Additional Guidelines:
        1. Always validate queries against user intent:
           - Prioritize **relevance and accuracy**.
           - Use domain-specific filtering for improved results (e.g., filtering by `pocategory_id` for purchase order categories).
        2. Incorporate prompt optimization techniques:
           - Break down **complex questions** into smaller SQL components to ensure accuracy.
           - Apply **logical conditions** (e.g., combining multiple filters using `AND` or `OR`) for precise results.
        3. Handle ambiguity:
           - Clarify the query if needed.
           - Make reasonable assumptions based on the schema and metadata.
        4. Optimize performance:
           - Use indexed columns in filtering conditions to speed up queries.
           - Aggregate results when large datasets are involved (e.g., using `SUM()`, `AVG()`, `GROUP BY`).
        
        5. Present answers effectively:
           - Use **Markdown** tables with proper column headers and alignments.
           - Provide **concise summaries** when large datasets are returned.

        6. For handling big result data:
           - The result is too large to display. Please refine your query or use filters to reduce the result size to show the `top ten` results only.

        """
        
        # Create the prompt and messages
        human_message = HumanMessagePromptTemplate.from_template("{input}").format(input=userinput)
        messages = [
            SystemMessage(content=prefix),
            human_message,
            AIMessage(content=suffix),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, prompt=prompt)

        # Execute the query
        response = agent_executor.invoke(f"Now answer this query: {userinput}")["output"]
        return {"response": response}
    except Exception as e:
        logging.error("Error handling query:", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

# Basic endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}

# Start the scheduler on app startup
@app.on_event("startup")
async def startup():
    scheduler.start()

# Shutdown the scheduler on app shutdown
@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()
