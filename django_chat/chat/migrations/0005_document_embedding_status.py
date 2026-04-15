from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0004_chatsession_user_document_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='embedding_status',
            field=models.CharField(
                choices=[('pending', 'Pending'), ('done', 'Done'), ('failed', 'Failed')],
                default='done',   # existing rows are already embedded, mark them done
                max_length=10,
            ),
        ),
    ]
