# Classification Project**

**Project Objectives**
We here at TELCO Communications™ take customer satisfaction to heart. We are always looking to improve our customer's experience with our company. Today, we would like to explore the possible drivers of our customer churn rate. The following Jupyter Notebook offers in depth exploration, analysis, and models to project how drivers affect our customer experience and ultimate retention.

**Business Goals**
Identify drivers for customer churn at TELCO.
Construct a ML classification model that accurately predicts customer churn.
create CSV file of these predictions.
Deliverables
a Jupyter Notebook Report showing process and analysis with the goal of finding drivers for customer churn.

a README.md file containing the project description with goals, a data dictionary, project planning, key findings, recommendations, and takeaways from your project.

a CSV file with customer_id, probability of churn, and prediction of churn. (1=churn, 0=not_churn)
individual modules, .py files, that hold your functions to acquire and prepare your data.

**Key Findings**:
* There appears to be a dependency on churn and month-to-month customer contracts.
* Month to month customers rate of churn higher than that of one-year contract customers
* Fiber customers have significantly higher monthly charges than customers without fiber octic.

# Reccomendations
- I would reccomend that the company should offer an incentive for customers on monnth-to-month contracts(especially those utilizing the fiber_optic) an incentive to either switch to a one year-plan or reduce prices for those customers on the month to month plans that have established some form of tenure with Telco.





## Data Dictionary

Name | Description | Type
:---: | :---: | :---:
senior_citizen | Indicates if customer is a senior citizen | int
tenure | Months customer has subscribed to service | int
monthly_charges | Dollar cost per month | float
total_charges | Dollar cost accumulated during tenure | float
internet_extras | Indicates if customer pays for internet add-ons | int
streaming_entertainment | Indicates if customer has streaming movies or tv | int
family | Indicates if customer has dependents or partner | int
gender_Male | Indicates if customer identifies as male | int
phone_service_Yes | Indicates if customer has at least 1 phone line | int
paperless_billing_Yes | Indicates if customer uses paperless billing | int
churn_Yes | Indicates if customer has left the company | int
contract_type_Month-to-month | Indicates if customer pays on a monthly basis | int
contract_type_One_year | Indicates if customer pays annually | int
contract_type_Two_year | Indicates if customer pays bi-annually | int
internet_service_type_DSL | Indicates if customer has DSL internet | int
internet_service_type_Fiber_optic | Indicates if customer has fiber optic internet | int
internet_service_type_None | Indicates if customer does not have internet | int
payment_type_Bank_transfer | Indicates if customer pays using a bank account | int
payment_type_Credit_card | Indicates if customer pays using a credit card | int
payment_type_Electronic_check | Indicates if customer pays using e-check | int
payment_type_Mailed_check | Indicates if customer pays using paper check | int
