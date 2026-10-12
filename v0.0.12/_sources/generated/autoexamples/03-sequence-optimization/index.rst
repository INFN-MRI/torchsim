

.. _sphx_glr_generated_autoexamples_03-sequence-optimization:

Applications: sequence design
-----------------------------

Choose sequence parameters by gradient descent through the simulation: the
refocusing angles of an echo train, the flip angles of a relaxometry
protocol, and the samples of an RF pulse.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example you design the refocusing flip angles of a fast spin echo for image quality rather than for precision: first one echo train, then a whole segmented 3D protocol in which each shot has its own repetition time, echo train length and angles.">

.. only:: html

  .. image:: /generated/autoexamples/03-sequence-optimization/images/thumb/sphx_glr_01-echo-train-design_thumb.png
    :alt:

  :doc:`/generated/autoexamples/03-sequence-optimization/01-echo-train-design`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Echo train design for sharpness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example you choose the flip angles of a DESPOT protocol so that a joint fit of T1 and T2 is as precise as possible [1]_. DESPOT estimates T1 from spoiled gradient-echo scans at several flip angles and T2 from balanced SSFP scans. A joint fit uses all the data for both parameters.">

.. only:: html

  .. image:: /generated/autoexamples/03-sequence-optimization/images/thumb/sphx_glr_02-joint-relaxometry_thumb.png
    :alt:

  :doc:`/generated/autoexamples/03-sequence-optimization/02-joint-relaxometry`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint relaxometry protocol design</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example you design the samples of a slice-selective RF pulse by gradient descent through a Bloch simulation of what they do [1]_. Across a body at 3 T the transmit field B1 varies by about a fifth either way, and the flip angle inside the slice varies with it. You reshape a 90 degree excitation so that it stays close to 90 degrees over that range.">

.. only:: html

  .. image:: /generated/autoexamples/03-sequence-optimization/images/thumb/sphx_glr_03-rf-pulse-design_thumb.png
    :alt:

  :doc:`/generated/autoexamples/03-sequence-optimization/03-rf-pulse-design`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">RF pulse design by optimal control</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/autoexamples/03-sequence-optimization/01-echo-train-design
   /generated/autoexamples/03-sequence-optimization/02-joint-relaxometry
   /generated/autoexamples/03-sequence-optimization/03-rf-pulse-design

