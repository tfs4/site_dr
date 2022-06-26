from django import forms
from .models import Upload

CHOICES = [('1', 'First'), ('2', 'Second')]

class UploadForm(forms.ModelForm):


    #Type_classification = forms.ChoiceField(choices=CHOICES)

    class Meta:
        model = Upload
        fields = ('image',)

