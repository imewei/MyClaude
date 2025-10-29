
document.addEventListener('DOMContentLoaded', () => {
    console.log('Page loaded');

    const button = document.querySelector('button');
    button?.addEventListener('click', () => {
        alert('Button clicked!');
    });
});
