from flask import Flask, render_template, request, redirect, flash, url_for
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

with open('credit_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

score_map = {
    "Poor": ("Poor", "score-poor", "Your credit score is low. Focus on reducing debt and making timely payments."),
    "Standard": ("Standard", "score-standard", "You're doing okay. Improve your score by paying bills on time and reducing debt."),
    "Good": ("Good", "score-good", "Excellent score! Keep managing your finances wisely.")
}

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['annual_income']),
                float(request.form['monthly_salary']),
                float(request.form['num_bank_accounts']),
                float(request.form['num_credit_cards']),
                float(request.form['interest_rate']),
                float(request.form['num_loans']),
                float(request.form['avg_days_delayed']),
                float(request.form['num_delayed_payments']),
                float(request.form['changed_credit_limit']),
                float(request.form['num_credit_inquiries']),
                float(request.form['outstanding_debt']),
                float(request.form['credit_utilization_ratio']),
                float(request.form['credit_history_age']),
                float(request.form['total_emi']),
                float(request.form['amount_invested']),
                float(request.form['monthly_balance'])
            ]

            input_array = np.array([features])

            numeric_pred = model.predict(input_array)[0]

            label = le.inverse_transform([numeric_pred])[0]

            label, score_class, score_description = score_map.get(label, ("Unknown", "", "No description available."))

            return render_template('index.html',
                                   prediction=label,
                                   score_class=score_class,
                                   score_description=score_description)

        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "error")
            return redirect(url_for('predict'))

    return render_template('index.html', prediction=None)   

if __name__ == '__main__':
    app.run(debug=True)
