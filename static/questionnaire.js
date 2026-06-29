document.getElementById('discoveryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    submitBtn.innerText = 'Saving...';
    submitBtn.disabled = true;

    // Collect answers
    const answers = {};
    for (let i = 1; i <= 17; i++) {
        answers[`q${i}`] = document.getElementById(`q${i}`).value;
    }

    const payload = {
        interviewee_name: document.getElementById('interviewee_name').value,
        company: document.getElementById('company').value,
        answers: answers
    };

    try {
        const response = await fetch('/api/responses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            document.getElementById('successMsg').style.display = 'block';
            document.getElementById('discoveryForm').reset();
            setTimeout(() => {
                document.getElementById('successMsg').style.display = 'none';
            }, 5000);
        } else {
            alert('Failed to save to database. Check console logs.');
        }
    } catch (error) {
        console.error('Error saving response:', error);
        alert('Network error while saving response.');
    } finally {
        submitBtn.innerText = 'Save to PostgreSQL';
        submitBtn.disabled = false;
    }
});
