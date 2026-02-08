"""
Projector module for mapping between student and teacher dimensions.
"""

import torch.nn as nn


class Projector(nn.Module):
    """
    Projects student model representations to teacher model dimensions.
    """
    
    def __init__(self, d_student, d_teacher):
        super().__init__()
        self.d_student = d_student
        self.d_teacher = d_teacher
        self.projector = nn.Linear(self.d_student, self.d_teacher) 
        self.ln = nn.LayerNorm(self.d_student)
    
    def forward(self, x):
        return self.projector(self.ln(x))