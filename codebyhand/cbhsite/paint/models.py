from django.db import models

# Create your models here.
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published")

    def __str__(self):
        return self.question_text


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)

    def __str__(self):
        return self.choice_text


class Point(models.Model):
    stroke_id = models.IntegerField()
    pix_id = models.IntegerField()
    x = models.PositiveIntegerField()
    y = models.PositiveIntegerField()
    img_fn = models.FileField(max_length=256)

    def __str__(self):
        return self.stroke_id, self.x, self.y


class Strokes(models.Model):
    stroke_id = models.ForeignKey(Point, on_delete=models.CASCADE)
    label = models.CharField(null=True)
    img_fn = models.FileField(max_length=256, unique=True)
    pub_date = models.DateTimeField("stroke_time")

    def __str__(self):
        return self.stroke_id, self.img_fn, self.pub_date
