from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def user_view_dataset(request):
    import pandas as pd
    from django.conf import settings
    path = settings.MEDIA_ROOT + "\\" + 'All_Emails.xlsx'
    df = pd.read_excel(path)
    df = df.to_html
    return render(request, 'users/dataset_view.html', {'data': df})


def user_classification_emails(request):
    from .utility import processAlgorithms
    accuracy, precision, recall = processAlgorithms.start_process()
    return render(request, 'users/view_classification_result.html',
                  {'accuracy': accuracy, 'precision': precision, 'recall': recall})


def user_view_cleaned_data(request):
    from .utility import cleanedData
    df = cleanedData.start_cleaning_process();
    df = df.to_html
    return render(request, 'users/view_cleaned_data.html', {'data': df})


def user_spam_predict(request):
    if request.method == 'POST':
        mailBody = request.POST.get("mailbody")
        from .utility import checkSpamorNot
        result = checkSpamorNot.test_mail_is_spam_or_not(mailBody)
        return render(request, 'users/test_results.html', {'mail': mailBody, 'result': result})
    else:
        return render(request, 'users/test_results.html', {})
